"""Import the Withings history into long-term statistics.

The sensors only hold the latest value and record it when Home Assistant
receives it. The Withings API keeps the full history with the time each value
was measured, so it is imported as external statistics: measurements and sleep
summaries as hourly mean/min/max, daily activity totals as sums.
"""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from aiowithings import (
    Activity,
    MeasurementAttribution,
    MeasurementGroup,
    MeasurementGroupCategory,
    MeasurementPosition,
    WithingsClient,
    WithingsError,
)

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    STATISTIC_UNIT_TO_UNIT_CONVERTER,
    async_add_external_statistics,
    statistics_during_period,
)
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.storage import Store
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util

from .const import DOMAIN, LOGGER
from .coordinator import SLEEP_SUMMARY_DATA_FIELDS, WithingsDataUpdateCoordinator
from .sensor import (
    ACTIVITY_SENSORS,
    SLEEP_SENSORS,
    WithingsActivitySensorEntityDescription,
    WithingsMeasurementSensorEntityDescription,
    WithingsSleepSensorEntityDescription,
    get_measurement_description,
)

if TYPE_CHECKING:
    from . import WithingsConfigEntry

STORAGE_VERSION = 1

MEASUREMENTS = "measurements"
SLEEP = "sleep"
ACTIVITY = "activity"

# aiowithings does not follow the API's pagination, so history is requested
# in windows that stay well below the size of a page.
MEASUREMENT_WINDOW = timedelta(days=7)
DAILY_WINDOW = timedelta(days=30)
# Spread the history requests to stay below the Withings API rate limit.
REQUEST_DELAY = 1
# Pause between import runs to coalesce webhook bursts.
IMPORT_COOLDOWN = 60
# Used when no device reports when it was first used.
DEFAULT_HISTORY = timedelta(days=365)

IGNORED_ATTRIBUTIONS = {
    MeasurementAttribution.UNKNOWN,
    MeasurementAttribution.DEVICE_ENTRY_FOR_USER_AMBIGUOUS,
}
WHOLE_BODY_POSITIONS = {
    MeasurementPosition.WHOLE_BODY,
    MeasurementPosition.BETWEEN_LEGS,
}

type _Description = (
    WithingsActivitySensorEntityDescription
    | WithingsMeasurementSensorEntityDescription
    | WithingsSleepSensorEntityDescription
)


def _storage_key(entry: WithingsConfigEntry) -> str:
    return f"{DOMAIN}.{entry.entry_id}.statistics"


async def async_remove_statistics_store(
    hass: HomeAssistant, entry: WithingsConfigEntry
) -> None:
    """Remove the import progress of a removed entry."""
    await Store[dict[str, float]](
        hass, STORAGE_VERSION, _storage_key(entry)
    ).async_remove()


def _start_of_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


def _start_of_day(day: date) -> datetime:
    """Return the first full hour of a local day, statistics start on the hour."""
    start = dt_util.as_utc(dt_util.start_of_local_day(day))
    if start.minute:
        start = _start_of_hour(start) + timedelta(hours=1)
    return start


def _datetime_windows(
    start: datetime, end: datetime
) -> Iterator[tuple[datetime, datetime]]:
    while start < end:
        window_end = min(start + MEASUREMENT_WINDOW, end)
        yield start, window_end
        start = window_end


def _date_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    while start <= end:
        window_end = min(start + DAILY_WINDOW - timedelta(days=1), end)
        yield start, window_end
        start = window_end + timedelta(days=1)


def _activity_key(description: WithingsActivitySensorEntityDescription) -> str:
    return description.key.removesuffix("_today")


def _unit_and_converter(
    description: _Description,
) -> tuple[str | None, Callable[[float], float]]:
    """Return the unit to store a statistic in, and how to convert to it."""
    native_unit = description.native_unit_of_measurement
    unit = description.suggested_unit_of_measurement or native_unit
    if (
        unit == native_unit
        or (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(native_unit)) is None
    ):
        return native_unit, float
    return unit, converter.converter_factory(native_unit, unit)


class WithingsStatisticsImporter:
    """Import the Withings history of a user into external statistics."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: WithingsConfigEntry,
        client: WithingsClient,
    ) -> None:
        """Initialize the importer."""
        self._hass = hass
        self._entry = entry
        self._client = client
        self._store = Store[dict[str, float]](
            hass, STORAGE_VERSION, _storage_key(entry)
        )
        # Per category, when the last successful import started. The next
        # import asks Withings for everything modified since then.
        self._cursors: dict[str, float] = {}
        self._translations: dict[str, str] = {}
        self._importers: dict[str, Callable[[datetime | None], Awaitable[None]]] = {
            MEASUREMENTS: self._async_import_measurements,
            SLEEP: self._async_import_sleep,
            ACTIVITY: self._async_import_activity,
        }
        self._pending: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        # Imports of one category must not interleave, activity sums build on
        # the previous days.
        self._lock = asyncio.Lock()

    async def async_setup(self) -> None:
        """Load the import progress and the statistic names."""
        self._cursors = await self._store.async_load() or {}
        # Statistic names can't be translated, use the English sensor names.
        self._translations = {
            **await async_get_translations(self._hass, "en", "entity", {DOMAIN}),
            **await async_get_translations(
                self._hass, "en", "entity_component", {SENSOR_DOMAIN}
            ),
        }

    @callback
    def async_listen(
        self, category: str, coordinator: WithingsDataUpdateCoordinator[Any]
    ) -> CALLBACK_TYPE:
        """Import a category whenever its coordinator has fetched new data."""

        @callback
        def _async_coordinator_updated() -> None:
            if coordinator.last_update_success:
                self.async_schedule_import(category)

        return coordinator.async_add_listener(_async_coordinator_updated)

    @callback
    def async_schedule_import(self, category: str) -> None:
        """Schedule importing what changed in a category since the last import."""
        self._pending.add(category)
        if self._task is None or self._task.done():
            self._task = self._entry.async_create_background_task(
                self._hass,
                self._async_run(),
                f"Withings {self._entry.unique_id} statistics import",
            )

    async def _async_run(self) -> None:
        while self._pending:
            pending, self._pending = self._pending, set()
            for category, importer in self._importers.items():
                if category in pending:
                    await self._async_import(category, importer)
            await asyncio.sleep(IMPORT_COOLDOWN)

    async def _async_import(
        self,
        category: str,
        importer: Callable[[datetime | None], Awaitable[None]],
    ) -> None:
        started = dt_util.utcnow()
        cursor = self._cursors.get(category)
        try:
            async with self._lock:
                await importer(
                    None if cursor is None else dt_util.utc_from_timestamp(cursor)
                )
        except WithingsError as err:
            LOGGER.warning(
                "Failed to import the Withings %s history of %s: %s",
                category,
                self._entry.title,
                err,
            )
            return
        self._cursors[category] = started.timestamp()
        await self._store.async_save(self._cursors)

    async def async_import_period(
        self, categories: set[str], start: date, end: date
    ) -> None:
        """Import the history of a period again, also before the devices were used.

        Activity is imported up to today, the sums of the later days build on it.
        """
        async with self._lock:
            if MEASUREMENTS in categories:
                await self._async_import_measurements_between(
                    dt_util.as_utc(dt_util.start_of_local_day(start)),
                    min(_start_of_day(end + timedelta(days=1)), dt_util.utcnow()),
                )
            if SLEEP in categories:
                await self._async_import_sleep_between(start, end)
            if ACTIVITY in categories:
                await self._async_import_activity_from(start)

    def _history_start(self) -> datetime:
        """Return when the user started using their current devices."""
        return min(
            (
                device.first_session_date
                for device in self._entry.runtime_data.device_coordinator.data.values()
                if device.first_session_date is not None
            ),
            default=dt_util.utcnow() - DEFAULT_HISTORY,
        )

    async def _async_fetch_windows[_T: (date, datetime), _R](
        self,
        windows: Iterable[tuple[_T, _T]],
        fetch: Callable[[_T, _T], Awaitable[list[_R]]],
    ) -> list[_R]:
        results: list[_R] = []
        for index, (start, end) in enumerate(windows):
            if index:
                await asyncio.sleep(REQUEST_DELAY)
            results.extend(await fetch(start, end))
        return results

    async def _async_import_measurements(self, since: datetime | None) -> None:
        if since is None:
            start = self._history_start()
        else:
            if not (changed := await self._client.get_measurement_since(since)):
                return
            start = min(group.taken_at for group in changed)
        await self._async_import_measurements_between(start, dt_util.utcnow())

    async def _async_import_measurements_between(
        self, start: datetime, end: datetime
    ) -> None:
        # Re-read whole hours, their statistics are replaced as a whole.
        start = _start_of_hour(start)
        groups: list[MeasurementGroup] = await self._async_fetch_windows(
            _datetime_windows(start, end), self._client.get_measurement_in_period
        )

        descriptions: dict[str, WithingsMeasurementSensorEntityDescription] = {}
        values: defaultdict[str, defaultdict[datetime, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Groups on a window boundary are returned twice.
        for group in {
            (group.group_id, group.taken_at): group for group in groups
        }.values():
            if (
                group.attribution in IGNORED_ATTRIBUTIONS
                or group.category is MeasurementGroupCategory.USER_OBJECTIVES
            ):
                continue
            hour = _start_of_hour(group.taken_at)
            for measurement in group.measurements:
                position = (
                    None
                    if measurement.position in WHOLE_BODY_POSITIONS
                    else measurement.position
                )
                if (
                    description := get_measurement_description(
                        (measurement.measurement_type, position)
                    )
                ) is None:
                    continue
                descriptions[description.key] = description
                values[description.key][hour].append(measurement.value)

        for key, hours in values.items():
            self._async_add_mean_statistics(descriptions[key], key, hours)

    async def _async_import_sleep(self, since: datetime | None) -> None:
        if since is None:
            start = self._history_start().date()
        else:
            if not (
                changed := await self._client.get_sleep_summary_since(
                    since, SLEEP_SUMMARY_DATA_FIELDS
                )
            ):
                return
            # Include the previous day, so all summaries ending in the same
            # hour as a changed one are imported together.
            start = min(summary.date for summary in changed) - timedelta(days=1)
        await self._async_import_sleep_between(start, dt_util.now().date())

    async def _async_import_sleep_between(self, start: date, end: date) -> None:
        summaries = await self._async_fetch_windows(
            _date_windows(start, end),
            lambda start, end: self._client.get_sleep_summary_in_period(
                start, end, SLEEP_SUMMARY_DATA_FIELDS
            ),
        )

        values: defaultdict[str, defaultdict[datetime, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for summary in {
            (summary.start_date, summary.end_date): summary for summary in summaries
        }.values():
            hour = _start_of_hour(summary.end_date)
            for description in SLEEP_SENSORS:
                if (value := description.value_fn(summary)) is not None:
                    values[description.key][hour].append(float(value))

        for description in SLEEP_SENSORS:
            if hours := values.get(description.key):
                self._async_add_mean_statistics(description, description.key, hours)

    async def _async_import_activity(self, since: datetime | None) -> None:
        if since is None:
            start = self._history_start().date()
        else:
            if not (changed := await self._client.get_activities_since(since)):
                return
            start = min(activity.date for activity in changed)
        await self._async_import_activity_from(start)

    async def _async_import_activity_from(self, start: date) -> None:
        activities: dict[date, Activity] = {}
        for activity in await self._async_fetch_windows(
            _date_windows(start, dt_util.now().date()),
            self._client.get_activities_in_period,
        ):
            # Keep the first activity of a day, like the sensors do.
            activities.setdefault(activity.date, activity)
        if not activities:
            return

        # Every day from the first one on is re-imported, so the sums continue
        # from the last day before it.
        metadata = {
            description.key: self._metadata(
                description, _activity_key(description), has_sum=True
            )
            for description in ACTIVITY_SENSORS
        }
        sums = await self._async_get_last_sums(
            {meta["statistic_id"] for meta in metadata.values()},
            _start_of_day(min(activities)),
        )
        for description in ACTIVITY_SENSORS:
            meta = metadata[description.key]
            _, convert = _unit_and_converter(description)
            total = sums.get(meta["statistic_id"], 0.0)
            statistics: list[StatisticData] = []
            for day, activity in sorted(activities.items()):
                if (value := description.value_fn(activity)) is None:
                    continue
                state = convert(float(value))
                total += state
                statistics.append(
                    StatisticData(start=_start_of_day(day), state=state, sum=total)
                )
            async_add_external_statistics(self._hass, meta, statistics)

    async def _async_get_last_sums(
        self, statistic_ids: set[str], before: datetime
    ) -> dict[str, float]:
        """Return the sum of the last statistic before a time."""
        # Statistics added by a previous import may not be committed yet.
        await get_instance(self._hass).async_block_till_done()
        sums: dict[str, float] = {}
        # Look at the previous days first, they are usually there.
        for start in (before - timedelta(days=2), datetime.fromtimestamp(0, UTC)):
            if not (missing := statistic_ids - sums.keys()):
                break
            stats = await get_instance(self._hass).async_add_executor_job(
                statistics_during_period,
                self._hass,
                start,
                before,
                missing,
                "hour",
                None,
                {"sum"},
            )
            for statistic_id, rows in stats.items():
                if rows and (last_sum := rows[-1]["sum"]) is not None:
                    sums[statistic_id] = last_sum
        return sums

    @callback
    def _async_add_mean_statistics(
        self,
        description: _Description,
        key: str,
        hours: dict[datetime, list[float]],
    ) -> None:
        _, convert = _unit_and_converter(description)
        statistics: list[StatisticData] = []
        for hour, hour_values in sorted(hours.items()):
            converted = [convert(value) for value in hour_values]
            statistics.append(
                StatisticData(
                    start=hour,
                    mean=sum(converted) / len(converted),
                    min=min(converted),
                    max=max(converted),
                )
            )
        async_add_external_statistics(
            self._hass, self._metadata(description, key, has_sum=False), statistics
        )

    def _metadata(
        self, description: _Description, key: str, *, has_sum: bool
    ) -> StatisticMetaData:
        unit, _ = _unit_and_converter(description)
        converter = STATISTIC_UNIT_TO_UNIT_CONVERTER.get(unit)
        return StatisticMetaData(
            mean_type=StatisticMeanType.NONE
            if has_sum
            else StatisticMeanType.ARITHMETIC,
            has_sum=has_sum,
            name=f"{self._entry.title} {self._name(description, daily=has_sum)}",
            source=DOMAIN,
            statistic_id=f"{DOMAIN}:{self._entry.unique_id}_{key}",
            unit_class=converter.UNIT_CLASS if converter else None,
            unit_of_measurement=unit,
        )

    def _name(self, description: _Description, *, daily: bool) -> str:
        name: str | None = None
        if description.translation_key:
            name = self._translations.get(
                f"component.{DOMAIN}.entity.sensor.{description.translation_key}.name"
            )
        elif description.device_class:
            name = self._translations.get(
                f"component.{SENSOR_DOMAIN}.entity_component."
                f"{description.device_class}.name"
            )
        if name is None:
            return description.key
        # The activity sensors are named after today, the statistics hold
        # every day.
        return name.removesuffix(" today") if daily else name
