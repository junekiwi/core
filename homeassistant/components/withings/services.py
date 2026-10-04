"""Services for the Withings integration."""

from datetime import date
from typing import TYPE_CHECKING

from aiowithings import WithingsError
import voluptuous as vol

from homeassistant.const import ATTR_CONFIG_ENTRY_ID
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, service
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .statistics import ACTIVITY, MEASUREMENTS, SLEEP

if TYPE_CHECKING:
    from . import WithingsConfigEntry

SERVICE_IMPORT_HISTORY = "import_history"

ATTR_START_DATE = "start_date"
ATTR_END_DATE = "end_date"
ATTR_DATA_TYPES = "data_types"

DATA_TYPES = [MEASUREMENTS, SLEEP, ACTIVITY]

IMPORT_HISTORY_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string,
        vol.Required(ATTR_START_DATE): cv.date,
        vol.Optional(ATTR_END_DATE): cv.date,
        vol.Optional(ATTR_DATA_TYPES, default=DATA_TYPES): vol.All(
            cv.ensure_list, [vol.In(DATA_TYPES)]
        ),
    }
)


async def _async_import_history(call: ServiceCall) -> None:
    """Import the Withings history of a period into the statistics."""
    entry: WithingsConfigEntry = service.async_get_config_entry(
        call.hass, DOMAIN, call.data[ATTR_CONFIG_ENTRY_ID]
    )
    today = dt_util.now().date()
    start: date = call.data[ATTR_START_DATE]
    end: date = min(call.data.get(ATTR_END_DATE, today), today)
    if start > end:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="invalid_import_period"
        )
    try:
        await entry.runtime_data.statistics_importer.async_import_period(
            set(call.data[ATTR_DATA_TYPES]), start, end
        )
    except WithingsError as err:
        raise HomeAssistantError(
            translation_domain=DOMAIN,
            translation_key="import_history_failed",
            translation_placeholders={"error": str(err)},
        ) from err


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register the Withings services."""
    # Overwrites long-term statistics.
    service.async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_IMPORT_HISTORY,
        _async_import_history,
        schema=IMPORT_HISTORY_SCHEMA,
    )
