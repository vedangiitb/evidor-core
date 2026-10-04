"""Current-time built-in tool."""

from datetime import datetime, timezone as datetime_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..core.tools import tool


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time as an ISO 8601 string.

    Args:
        timezone: IANA timezone name, such as 'UTC' or 'America/New_York'.
    """
    try:
        zone = datetime_timezone.utc if timezone.upper() in {"UTC", "ETC/UTC", "Z"} else ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone: {timezone}") from exc
    return datetime.now(zone).isoformat()
