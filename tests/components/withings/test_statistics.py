"""Tests for the Withings statistics import."""

from collections.abc import Generator, Iterable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from aiowithings import (
    Activity,
    Measurement,
    MeasurementAttribution,
    MeasurementGroup,
    MeasurementGroupCategory,
    MeasurementPosition,
    MeasurementType,
    NotificationCategory,
    SleepSummary,
    WithingsError,
)
from freezegun.api import FrozenDateTimeFactory
import pytest

from homeassistant.components.recorder.statistics import (
    list_statistic_ids,
    statistics_during_period,
)
from homeassistant.components.withings.const import DOMAIN
from homeassistant.const import ATTR_CONFIG_ENTRY_ID
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

from . import (
    call_webhook,
    load_activity_fixture,
    load_device_fixture,
    load_sleep_fixture,
    prepare_webhook_setup,
    setup_integration,
)
from .conftest import USER_ID, WEBHOOK_ID

from tests.common import MockConfigEntry
from tests.components.recorder.common import async_wait_recording_done
from tests.typing import ClientSessionGenerator

NOW = datetime(2023, 10, 21, 12, 0, tzinfo=UTC)
FIRST_USE = datetime(2023, 10, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def disable_statistics_import() -> Generator[None]:
    """Run the statistics import, without delays."""
    with (
        patch("homeassistant.components.withings.statistics.REQUEST_DELAY", 0),
        patch("homeassistant.components.withings.statistics.IMPORT_COOLDOWN", 0),
    ):
        yield


@pytest.fixture(autouse=True)
def frozen_time(freezer: FrozenDateTimeFactory) -> None:
    """Freeze the time after the test data."""
    freezer.move_to(NOW)


def _group(
    group_id: int,
    taken_at: datetime,
    measurements: list[Measurement],
    *,
    attribution: MeasurementAttribution = MeasurementAttribution.DEVICE_ENTRY_FOR_USER,
    category: MeasurementGroupCategory = MeasurementGroupCategory.REAL,
) -> MeasurementGroup:
    return MeasurementGroup(
        group_id=group_id,
        attribution=attribution,
        taken_at=taken_at,
        stored_at=taken_at,
        updated_at=taken_at,
        category=category,
        device_id="scale",
        hashed_device_id="scale",
        measurements=measurements,
    )


def _weight(value: float) -> Measurement:
    return Measurement(measurement_type=MeasurementType.WEIGHT, value=value)


def _serve_history(
    withings: AsyncMock,
    *,
    measurements: Iterable[MeasurementGroup] = (),
    sleep: Iterable[SleepSummary] = (),
    activities: Iterable[Activity] = (),
) -> None:
    """Make the API return the given history, filtered on the requested period."""
    withings.get_devices.return_value = [
        replace(device, first_session_date=FIRST_USE)
        for device in load_device_fixture()
    ]
    withings.get_measurement_in_period.side_effect = lambda start, end, *_: [
        group for group in measurements if start <= group.taken_at <= end
    ]
    withings.get_sleep_summary_in_period.side_effect = lambda start, end, *_: [
        summary for summary in sleep if start <= summary.date <= end
    ]
    withings.get_activities_in_period.side_effect = lambda start, end, *_: [
        activity for activity in activities if start <= activity.date <= end
    ]


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    await setup_integration(hass, entry, enable_webhooks=False)
    await _wait_for_import(hass)


async def _wait_for_import(hass: HomeAssistant) -> None:
    await hass.async_block_till_done(wait_background_tasks=True)
    await async_wait_recording_done(hass)


async def _statistics(
    hass: HomeAssistant, key: str, types: set[Any]
) -> list[dict[str, Any]]:
    statistic_id = f"{DOMAIN}:{USER_ID}_{key}"
    stats = await hass.async_add_executor_job(
        statistics_during_period,
        hass,
        datetime(2000, 1, 1, tzinfo=UTC),
        None,
        {statistic_id},
        "hour",
        None,
        types,
    )
    return [
        {"start": datetime.fromtimestamp(row["start"], UTC)}
        | {stat_type: row[stat_type] for stat_type in types}
        for row in stats.get(statistic_id, [])
    ]


async def _metadata(hass: HomeAssistant, key: str) -> dict[str, Any]:
    statistic_id = f"{DOMAIN}:{USER_ID}_{key}"
    (metadata,) = await hass.async_add_executor_job(
        list_statistic_ids, hass, {statistic_id}
    )
    return metadata


async def test_import_measurements(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test the measurement history is imported per hour."""
    _serve_history(
        withings,
        measurements=[
            _group(1, datetime(2023, 10, 2, 7, 10, tzinfo=UTC), [_weight(70)]),
            _group(
                2,
                datetime(2023, 10, 2, 7, 40, tzinfo=UTC),
                [
                    _weight(72),
                    Measurement(
                        measurement_type=MeasurementType.MUSCLE_MASS_FOR_SEGMENTS,
                        value=3.5,
                        position=MeasurementPosition.LEFT_ARM,
                    ),
                    Measurement(
                        measurement_type=MeasurementType.FAT_FREE_MASS,
                        value=55,
                        position=MeasurementPosition.WHOLE_BODY,
                    ),
                ],
            ),
            # Ignored like the sensors do.
            _group(
                3,
                datetime(2023, 10, 3, 7, 0, tzinfo=UTC),
                [_weight(100)],
                attribution=MeasurementAttribution.DEVICE_ENTRY_FOR_USER_AMBIGUOUS,
            ),
            _group(
                4,
                datetime(2023, 10, 4, 7, 0, tzinfo=UTC),
                [_weight(60)],
                category=MeasurementGroupCategory.USER_OBJECTIVES,
            ),
            # On the boundary of two requested periods.
            _group(5, datetime(2023, 10, 8, tzinfo=UTC), [_weight(68)]),
            _group(6, datetime(2023, 10, 20, 8, 5, tzinfo=UTC), [_weight(69)]),
        ],
    )

    await _setup(hass, polling_config_entry)

    assert await _statistics(hass, "weight_kg", {"mean", "min", "max"}) == [
        {
            "start": datetime(2023, 10, 2, 7, tzinfo=UTC),
            "mean": 71.0,
            "min": 70.0,
            "max": 72.0,
        },
        {
            "start": datetime(2023, 10, 8, tzinfo=UTC),
            "mean": 68.0,
            "min": 68.0,
            "max": 68.0,
        },
        {
            "start": datetime(2023, 10, 20, 8, tzinfo=UTC),
            "mean": 69.0,
            "min": 69.0,
            "max": 69.0,
        },
    ]
    assert await _statistics(hass, "muscle_mass_for_segments_left_arm", {"mean"}) == [
        {"start": datetime(2023, 10, 2, 7, tzinfo=UTC), "mean": 3.5}
    ]
    assert await _statistics(hass, "fat_free_mass_kg", {"mean"}) == [
        {"start": datetime(2023, 10, 2, 7, tzinfo=UTC), "mean": 55.0}
    ]
    metadata = await _metadata(hass, "weight_kg")
    assert metadata["name"] == "henk Weight"
    assert metadata["source"] == DOMAIN
    assert metadata["unit_class"] == "mass"
    assert metadata["statistics_unit_of_measurement"] == "kg"
    assert metadata["has_sum"] is False

    # After the sensors' request, the history is requested a week at a time
    # from the first device use.
    assert [
        call.args for call in withings.get_measurement_in_period.call_args_list
    ] == [
        (NOW - timedelta(days=14), NOW),
        (FIRST_USE, datetime(2023, 10, 8, tzinfo=UTC)),
        (datetime(2023, 10, 8, tzinfo=UTC), datetime(2023, 10, 15, tzinfo=UTC)),
        (datetime(2023, 10, 15, tzinfo=UTC), NOW),
    ]


async def test_import_sleep(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test sleep summaries are imported at the hour they ended."""
    summary = load_sleep_fixture()[0]
    _serve_history(
        withings,
        sleep=[
            replace(
                summary,
                date=date(2023, 10, 5),
                start_date=datetime(2023, 10, 4, 21, 30, tzinfo=UTC),
                end_date=datetime(2023, 10, 5, 6, 22, tzinfo=UTC),
                deep_sleep_duration=7200,
                sleep_score=90,
            ),
            replace(
                summary,
                date=date(2023, 10, 20),
                start_date=datetime(2023, 10, 19, 22, 0, tzinfo=UTC),
                end_date=datetime(2023, 10, 20, 5, 1, tzinfo=UTC),
                deep_sleep_duration=5400,
                sleep_score=None,
            ),
        ],
    )

    await _setup(hass, polling_config_entry)

    assert await _statistics(hass, "sleep_deep_duration_seconds", {"mean"}) == [
        {"start": datetime(2023, 10, 5, 6, tzinfo=UTC), "mean": 2.0},
        {"start": datetime(2023, 10, 20, 5, tzinfo=UTC), "mean": 1.5},
    ]
    assert await _statistics(hass, "sleep_score", {"mean"}) == [
        {"start": datetime(2023, 10, 5, 6, tzinfo=UTC), "mean": 90.0},
    ]
    metadata = await _metadata(hass, "sleep_deep_duration_seconds")
    assert metadata["name"] == "henk Deep sleep"
    assert metadata["statistics_unit_of_measurement"] == "h"
    assert metadata["unit_class"] == "duration"
    assert withings.get_sleep_summary_in_period.call_args_list[0].args[:2] == (
        date(2023, 10, 1),
        date(2023, 10, 21),
    )


async def test_import_activity(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test daily activity totals are imported as sums."""
    _serve_history(withings, activities=load_activity_fixture())

    await _setup(hass, polling_config_entry)

    steps = await _statistics(hass, "activity_steps", {"state", "sum"})
    assert len(steps) == 14
    # Days start at local midnight, US/Pacific.
    assert steps[:3] == [
        {"start": datetime(2023, 10, 8, 7, tzinfo=UTC), "state": 1892, "sum": 1892},
        {"start": datetime(2023, 10, 9, 7, tzinfo=UTC), "state": 2576, "sum": 4468},
        {"start": datetime(2023, 10, 10, 7, tzinfo=UTC), "state": 1827, "sum": 6295},
    ]
    assert steps[-1]["sum"] == sum(
        activity.steps for activity in load_activity_fixture()
    )
    metadata = await _metadata(hass, "activity_steps")
    assert metadata["name"] == "henk Steps"
    assert metadata["has_sum"] is True
    assert metadata["statistics_unit_of_measurement"] == "steps"
    assert (await _statistics(hass, "activity_soft_duration", {"state"}))[0] == {
        "start": datetime(2023, 10, 8, 7, tzinfo=UTC),
        "state": pytest.approx(4981 / 60),
    }
    assert (await _metadata(hass, "activity_soft_duration"))["name"] == (
        "henk Soft activity"
    )


async def test_webhook_imports_changes(
    hass: HomeAssistant,
    withings: AsyncMock,
    webhook_config_entry: MockConfigEntry,
    hass_client_no_auth: ClientSessionGenerator,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Test a webhook imports what changed since the previous import."""
    activities = load_activity_fixture()[:-1]
    measurements = [
        _group(1, datetime(2023, 10, 21, 11, 10, tzinfo=UTC), [_weight(70)])
    ]
    _serve_history(withings, measurements=measurements, activities=activities)
    await setup_integration(hass, webhook_config_entry)
    await _wait_for_import(hass)
    await prepare_webhook_setup(hass, freezer)
    steps = await _statistics(hass, "activity_steps", {"state", "sum"})
    assert len(steps) == 13
    sum_before_last_day = steps[-2]["sum"]

    # The last day synced again with more steps, and a new day started.
    changed = [
        replace(activities[-1], steps=5000),
        replace(activities[-1], date=date(2023, 10, 21), steps=100),
    ]
    activities[-1:] = changed
    withings.get_activities_since.return_value = changed
    client = await hass_client_no_auth()
    resp = await call_webhook(
        hass,
        WEBHOOK_ID,
        {"userid": USER_ID, "appli": NotificationCategory.ACTIVITY},
        client,
    )
    assert resp.message_code == 0
    await _wait_for_import(hass)

    withings.get_activities_since.assert_called_with(NOW)
    steps = await _statistics(hass, "activity_steps", {"state", "sum"})
    assert len(steps) == 14
    assert steps[-2:] == [
        {
            "start": datetime(2023, 10, 20, 7, tzinfo=UTC),
            "state": 5000,
            "sum": sum_before_last_day + 5000,
        },
        {
            "start": datetime(2023, 10, 21, 7, tzinfo=UTC),
            "state": 100,
            "sum": sum_before_last_day + 5100,
        },
    ]

    # A second weigh-in in the same hour, the hour is imported as a whole.
    new_group = _group(2, datetime(2023, 10, 21, 11, 50, tzinfo=UTC), [_weight(72)])
    measurements.append(new_group)
    withings.get_measurement_since.return_value = [new_group]
    resp = await call_webhook(
        hass,
        WEBHOOK_ID,
        {"userid": USER_ID, "appli": NotificationCategory.WEIGHT},
        client,
    )
    assert resp.message_code == 0
    await _wait_for_import(hass)

    assert await _statistics(hass, "weight_kg", {"mean", "min", "max"}) == [
        {
            "start": datetime(2023, 10, 21, 11, tzinfo=UTC),
            "mean": 71.0,
            "min": 70.0,
            "max": 72.0,
        }
    ]


async def test_progress_is_stored(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """Test a restart only imports what changed since the previous import."""
    storage_key = f"{DOMAIN}.{polling_config_entry.entry_id}.statistics"
    previous_import = NOW - timedelta(hours=5)
    hass_storage[storage_key] = {
        "version": 1,
        "data": {
            "measurements": previous_import.timestamp(),
            "sleep": previous_import.timestamp(),
            "activity": previous_import.timestamp(),
        },
    }
    _serve_history(withings, activities=load_activity_fixture())
    withings.get_measurement_since.return_value = []
    withings.get_sleep_summary_since.return_value = []
    withings.get_activities_since.return_value = []

    await _setup(hass, polling_config_entry)

    withings.get_measurement_since.assert_called_once_with(previous_import)
    withings.get_activities_since.assert_called_once_with(previous_import)
    withings.get_sleep_summary_in_period.assert_not_called()
    assert await _statistics(hass, "activity_steps", {"sum"}) == []
    assert hass_storage[storage_key]["data"] == {
        "measurements": NOW.timestamp(),
        "sleep": NOW.timestamp(),
        "activity": NOW.timestamp(),
    }


async def test_failed_import_is_retried(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
    freezer: FrozenDateTimeFactory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a failed import keeps its progress and is retried on the next update."""
    summary = replace(
        load_sleep_fixture()[0],
        date=date(2023, 10, 5),
        end_date=datetime(2023, 10, 5, 6, 22, tzinfo=UTC),
        deep_sleep_duration=3600,
    )
    _serve_history(withings, sleep=[summary])
    withings.get_sleep_summary_in_period.side_effect = WithingsError("Too bad")

    await _setup(hass, polling_config_entry)

    assert "Failed to import the Withings sleep history of henk: Too bad" in (
        caplog.text
    )
    storage_key = f"{DOMAIN}.{polling_config_entry.entry_id}.statistics"
    assert "sleep" not in hass_storage[storage_key]["data"]
    assert "measurements" in hass_storage[storage_key]["data"]

    _serve_history(withings, sleep=[summary])
    await polling_config_entry.runtime_data.sleep_coordinator.async_refresh()
    await _wait_for_import(hass)

    assert await _statistics(hass, "sleep_deep_duration_seconds", {"mean"}) == [
        {"start": datetime(2023, 10, 5, 6, tzinfo=UTC), "mean": 1.0}
    ]
    assert "sleep" in hass_storage[storage_key]["data"]


async def test_remove_entry(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """Test the import progress is removed with the entry."""
    _serve_history(withings)
    await _setup(hass, polling_config_entry)
    storage_key = f"{DOMAIN}.{polling_config_entry.entry_id}.statistics"
    assert storage_key in hass_storage

    await hass.config_entries.async_remove(polling_config_entry.entry_id)
    await hass.async_block_till_done()

    assert storage_key not in hass_storage


async def test_import_history_service(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test importing a period from before the devices were used."""
    activities = load_activity_fixture()
    summary = load_sleep_fixture()[0]
    _serve_history(
        withings,
        measurements=[
            _group(1, datetime(2023, 9, 10, 8, 30, tzinfo=UTC), [_weight(80)]),
            _group(2, datetime(2023, 10, 20, 8, 5, tzinfo=UTC), [_weight(69)]),
        ],
        sleep=[
            replace(
                summary,
                date=date(2023, 9, 10),
                end_date=datetime(2023, 9, 10, 6, 10, tzinfo=UTC),
                deep_sleep_duration=3600,
            )
        ],
        activities=[
            replace(activities[0], date=date(2023, 9, 10), steps=1000),
            replace(activities[0], date=date(2023, 9, 11), steps=500),
            *activities,
        ],
    )
    await _setup(hass, polling_config_entry)
    assert len(await _statistics(hass, "weight_kg", {"mean"})) == 1
    steps = await _statistics(hass, "activity_steps", {"sum"})
    assert steps[0] == {"start": datetime(2023, 10, 8, 7, tzinfo=UTC), "sum": 1892}

    await hass.services.async_call(
        DOMAIN,
        "import_history",
        {
            ATTR_CONFIG_ENTRY_ID: polling_config_entry.entry_id,
            "start_date": "2023-09-01",
            "end_date": "2023-09-30",
        },
        blocking=True,
    )
    await async_wait_recording_done(hass)

    assert await _statistics(hass, "weight_kg", {"mean"}) == [
        {"start": datetime(2023, 9, 10, 8, tzinfo=UTC), "mean": 80.0},
        {"start": datetime(2023, 10, 20, 8, tzinfo=UTC), "mean": 69.0},
    ]
    assert await _statistics(hass, "sleep_deep_duration_seconds", {"mean"}) == [
        {"start": datetime(2023, 9, 10, 6, tzinfo=UTC), "mean": 1.0}
    ]
    # The activity after the period is summed again on top of it.
    steps = await _statistics(hass, "activity_steps", {"sum"})
    assert steps[:3] == [
        {"start": datetime(2023, 9, 10, 7, tzinfo=UTC), "sum": 1000},
        {"start": datetime(2023, 9, 11, 7, tzinfo=UTC), "sum": 1500},
        {"start": datetime(2023, 10, 8, 7, tzinfo=UTC), "sum": 3392},
    ]
    assert steps[-1]["sum"] == 1500 + sum(activity.steps for activity in activities)
    # Measurements are requested from the local start of the first day up to
    # the end of the last one.
    assert [
        call.args for call in withings.get_measurement_in_period.call_args_list[-5:]
    ] == [
        (datetime(2023, 9, 1, 7, tzinfo=UTC), datetime(2023, 9, 8, 7, tzinfo=UTC)),
        (datetime(2023, 9, 8, 7, tzinfo=UTC), datetime(2023, 9, 15, 7, tzinfo=UTC)),
        (datetime(2023, 9, 15, 7, tzinfo=UTC), datetime(2023, 9, 22, 7, tzinfo=UTC)),
        (datetime(2023, 9, 22, 7, tzinfo=UTC), datetime(2023, 9, 29, 7, tzinfo=UTC)),
        (datetime(2023, 9, 29, 7, tzinfo=UTC), datetime(2023, 10, 1, 7, tzinfo=UTC)),
    ]


async def test_import_history_service_data_types(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test only the requested kinds of data are imported."""
    _serve_history(withings)
    await _setup(hass, polling_config_entry)
    withings.get_measurement_in_period.reset_mock()
    withings.get_activities_in_period.reset_mock()
    withings.get_sleep_summary_in_period.reset_mock()

    await hass.services.async_call(
        DOMAIN,
        "import_history",
        {
            ATTR_CONFIG_ENTRY_ID: polling_config_entry.entry_id,
            "start_date": "2023-09-01",
            "data_types": ["sleep"],
        },
        blocking=True,
    )

    withings.get_measurement_in_period.assert_not_called()
    withings.get_activities_in_period.assert_not_called()
    assert [
        call.args[:2] for call in withings.get_sleep_summary_in_period.call_args_list
    ] == [
        (date(2023, 9, 1), date(2023, 9, 30)),
        (date(2023, 10, 1), date(2023, 10, 21)),
    ]


async def test_import_history_service_invalid_period(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test a start after the end, or in the future, is refused."""
    _serve_history(withings)
    await _setup(hass, polling_config_entry)

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            "import_history",
            {
                ATTR_CONFIG_ENTRY_ID: polling_config_entry.entry_id,
                "start_date": "2023-10-22",
            },
            blocking=True,
        )
    assert exc_info.value.translation_key == "invalid_import_period"


async def test_import_history_service_failure(
    hass: HomeAssistant,
    withings: AsyncMock,
    polling_config_entry: MockConfigEntry,
) -> None:
    """Test an API error is reported to the caller."""
    _serve_history(withings)
    await _setup(hass, polling_config_entry)
    withings.get_sleep_summary_in_period.side_effect = WithingsError("Too bad")

    with pytest.raises(HomeAssistantError) as exc_info:
        await hass.services.async_call(
            DOMAIN,
            "import_history",
            {
                ATTR_CONFIG_ENTRY_ID: polling_config_entry.entry_id,
                "start_date": "2023-10-01",
                "data_types": ["sleep"],
            },
            blocking=True,
        )
    assert exc_info.value.translation_key == "import_history_failed"
    assert exc_info.value.translation_placeholders == {"error": "Too bad"}
