"""
MACRO Economic Event Calendar
Tracks high-impact events and issues no-trade windows.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Optional, Tuple


@dataclass
class EconomicEvent:
    name: str
    time: datetime
    currency: str       # "USD", "XAU", etc.
    impact: str         # "HIGH", "MEDIUM", "LOW"
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


class EconomicCalendar:
    """
    High-impact events that APHELION must avoid trading.
    
    Data source: manually maintained with option for ForexFactory feed.
    """

    HIGH_IMPACT_EVENTS = [
        "FOMC Rate Decision",
        "Federal Funds Rate",
        "Non-Farm Employment",
        "CPI m/m",
        "CPI y/y",
        "Core CPI m/m",
        "GDP q/q",
        "PPI m/m",
        "Powell Speech",
        "Fed Chair Press Conference",
        "Initial Jobless Claims",
        "ISM Manufacturing PMI",
        "Retail Sales m/m",
    ]

    def __init__(
        self,
        no_trade_before: timedelta = timedelta(minutes=30),
        no_trade_after: timedelta = timedelta(hours=1),
    ):
        self._before = no_trade_before
        self._after = no_trade_after
        self._events: List[EconomicEvent] = []

    def add_event(self, event: EconomicEvent) -> None:
        self._events.append(event)
        self._events.sort(key=lambda e: e.time)

    def add_events(self, events: List[EconomicEvent]) -> None:
        self._events.extend(events)
        self._events.sort(key=lambda e: e.time)

    def get_no_trade_windows(self, target_date: date) -> List[Tuple[datetime, datetime]]:
        """Return no-trade windows for a given date."""
        windows = []
        for event in self._events:
            if event.time.date() == target_date and event.impact == "HIGH":
                if event.currency in ("USD", "XAU", "ALL"):
                    start = event.time - self._before
                    end = event.time + self._after
                    windows.append((start, end))
        return windows

    def is_safe_to_trade(self, current_time: datetime) -> Tuple[bool, Optional[str]]:
        """Check if it's safe to trade right now."""
        for event in self._events:
            if event.impact != "HIGH":
                continue
            if event.currency not in ("USD", "XAU", "ALL"):
                continue

            before_start = event.time - self._before
            after_end = event.time + self._after

            if before_start <= current_time <= after_end:
                if current_time < event.time:
                    return False, f"Pre-event block: {event.name} at {event.time}"
                else:
                    return False, f"Post-event block: {event.name} at {event.time}"

        return True, None

    def get_next_event(self, current_time: datetime) -> Optional[EconomicEvent]:
        """Get the next upcoming high-impact event."""
        for event in self._events:
            if event.time > current_time and event.impact == "HIGH":
                return event
        return None

    def get_events_for_date(self, target_date: date) -> List[EconomicEvent]:
        return [e for e in self._events if e.time.date() == target_date]

    def clear(self) -> None:
        self._events.clear()
