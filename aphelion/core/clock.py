"""
APHELION Market Clock
Session detection, news calendar, trading hour management.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from aphelion.core.config import (
    Session, SessionWindow, SESSION_WINDOWS, SENTINEL,
    Timeframe,
)


class MarketClock:
    """Tracks market sessions, news events, and trading hours."""

    def __init__(self):
        self._news_calendar: list[dict] = []
        self._market_close_friday_utc = (21, 0)  # Friday 21:00 UTC
        self._market_open_sunday_utc = (22, 0)    # Sunday 22:00 UTC

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def current_session(self, dt: Optional[datetime] = None) -> Session:
        dt = dt or self.now_utc()
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute

        for window in SESSION_WINDOWS:
            start = window.open_hour * 60 + window.open_minute
            end = window.close_hour * 60 + window.close_minute
            if start <= time_minutes < end:
                return window.name

        return Session.DEAD_ZONE

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        weekday = dt.weekday()  # 0=Monday, 6=Sunday

        # Market closed: Friday 21:00 UTC → Sunday 22:00 UTC
        if weekday == 4:  # Friday
            close_time = dt.replace(hour=21, minute=0, second=0, microsecond=0)
            if dt >= close_time:
                return False
        elif weekday == 5:  # Saturday
            return False
        elif weekday == 6:  # Sunday
            open_time = dt.replace(hour=22, minute=0, second=0, microsecond=0)
            if dt < open_time:
                return False

        return True

    def is_trading_session(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        if not self.is_market_open(dt):
            return False
        session = self.current_session(dt)
        return session in (Session.LONDON, Session.NEW_YORK, Session.OVERLAP_LDN_NY)

    def minutes_to_session(self, target: Session, dt: Optional[datetime] = None) -> float:
        dt = dt or self.now_utc()
        current_minutes = dt.hour * 60 + dt.minute

        for window in SESSION_WINDOWS:
            if window.name == target:
                target_minutes = window.open_hour * 60 + window.open_minute
                diff = target_minutes - current_minutes
                if diff < 0:
                    diff += 24 * 60  # Next day
                return diff

        return float('inf')

    def minutes_to_close(self, dt: Optional[datetime] = None) -> float:
        dt = dt or self.now_utc()
        session = self.current_session(dt)
        current_minutes = dt.hour * 60 + dt.minute

        for window in SESSION_WINDOWS:
            if window.name == session:
                close_minutes = window.close_hour * 60 + window.close_minute
                diff = close_minutes - current_minutes
                return max(0, diff)

        return 0

    def is_friday_lockout(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        if dt.weekday() != 4:  # Not Friday
            return False

        close_time = dt.replace(hour=21, minute=0, second=0, microsecond=0)
        lockout_time = close_time - timedelta(minutes=SENTINEL.friday_close_lockout_minutes)
        return dt >= lockout_time

    def set_news_calendar(self, events: list[dict]) -> None:
        """Set news calendar. Each event: {'time': datetime, 'impact': 'HIGH'|'MED'|'LOW', 'name': str}"""
        self._news_calendar = sorted(events, key=lambda e: e["time"])

    def next_high_impact_news(self, dt: Optional[datetime] = None) -> Optional[dict]:
        dt = dt or self.now_utc()
        for event in self._news_calendar:
            if event["time"] > dt and event.get("impact") == "HIGH":
                return event
        return None

    def minutes_to_next_news(self, dt: Optional[datetime] = None) -> float:
        dt = dt or self.now_utc()
        event = self.next_high_impact_news(dt)
        if event is None:
            return float('inf')
        delta = (event["time"] - dt).total_seconds() / 60.0
        return delta

    def is_news_lockout(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        for event in self._news_calendar:
            if event.get("impact") != "HIGH":
                continue
            event_time = event["time"]
            pre_lockout = event_time - timedelta(minutes=SENTINEL.pre_news_lockout_minutes)
            post_lockout = event_time + timedelta(minutes=SENTINEL.post_news_lockout_minutes)
            if pre_lockout <= dt <= post_lockout:
                return True
        return False

    def last_high_impact_news_minutes(self, dt: Optional[datetime] = None) -> float:
        dt = dt or self.now_utc()
        for event in reversed(self._news_calendar):
            if event["time"] <= dt and event.get("impact") == "HIGH":
                return (dt - event["time"]).total_seconds() / 60.0
        return float('inf')

    def day_of_week(self, dt: Optional[datetime] = None) -> str:
        dt = dt or self.now_utc()
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        return days[dt.weekday()]

    def week_of_month(self, dt: Optional[datetime] = None) -> int:
        dt = dt or self.now_utc()
        return (dt.day - 1) // 7 + 1

    def is_month_end(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
        days_remaining = (next_month - dt).days
        return days_remaining <= 2

    def is_quarter_end(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        return dt.month in (3, 6, 9, 12) and self.is_month_end(dt)

    def session_features(self, dt: Optional[datetime] = None) -> dict:
        dt = dt or self.now_utc()
        return {
            "session": self.current_session(dt).name,
            "minutes_to_london_open": self.minutes_to_session(Session.LONDON, dt),
            "minutes_to_ny_open": self.minutes_to_session(Session.NEW_YORK, dt),
            "minutes_to_session_close": self.minutes_to_close(dt),
            "day_of_week": self.day_of_week(dt),
            "week_of_month": self.week_of_month(dt),
            "minutes_to_next_news": self.minutes_to_next_news(dt),
            "last_news_minutes_ago": self.last_high_impact_news_minutes(dt),
            "is_month_end": self.is_month_end(dt),
            "is_quarter_end": self.is_quarter_end(dt),
            "is_friday_lockout": self.is_friday_lockout(dt),
            "is_news_lockout": self.is_news_lockout(dt),
            "market_open": self.is_market_open(dt),
        }
