"""
APHELION Market Clock
Session detection, news calendar, trading hour management.
DST-aware for London/New York session shifts.
"""

import bisect
import math
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
        self._news_times: list[datetime] = []  # sorted high-impact times for bisect
        self._market_close_friday_utc = (21, 0)  # Friday 21:00 UTC
        self._market_open_sunday_utc = (22, 0)    # Sunday 22:00 UTC
        self._simulated_time: Optional[datetime] = None  # FIXED: backtest clock override
        self._dst_offset_london: int = 0   # 0 in winter, -60 in summer (sessions shift earlier)
        self._dst_offset_ny: int = 0       # 0 in winter, -60 in summer

    def set_dst_offsets(self, london_minutes: int = 0, ny_minutes: int = 0) -> None:
        """Set DST offsets for session window adjustments.
        In summer: London opens at 07:00 UTC (offset -60), NY opens 12:30 UTC (offset -60)."""
        self._dst_offset_london = london_minutes
        self._dst_offset_ny = ny_minutes

    def set_simulated_time(self, dt: Optional[datetime]) -> None:
        """Override now_utc() for deterministic backtesting."""
        self._simulated_time = dt

    def now_utc(self) -> datetime:
        # FIXED: Use simulated time when set (backtest mode)
        if self._simulated_time is not None:
            return self._simulated_time
        return datetime.now(timezone.utc)

    def current_session(self, dt: Optional[datetime] = None) -> Session:
        dt = dt or self.now_utc()
        hour = dt.hour
        minute = dt.minute
        time_minutes = hour * 60 + minute

        # v2: Apply DST offsets to session windows for correct detection year-round
        adjusted_windows = self._get_adjusted_windows()
        for window in adjusted_windows:
            start = window.open_hour * 60 + window.open_minute
            end = window.close_hour * 60 + window.close_minute
            if start <= time_minutes < end:
                return window.name

        return Session.DEAD_ZONE

    def _get_adjusted_windows(self) -> list[SessionWindow]:
        """Return session windows adjusted for current DST offsets."""
        adjusted = []
        for w in SESSION_WINDOWS:
            # London and Overlap sessions are affected by London DST
            if w.name in (Session.LONDON, Session.OVERLAP_LDN_NY):
                w = w.adjusted(self._dst_offset_london)
            # New York session is affected by NY DST
            elif w.name == Session.NEW_YORK:
                w = w.adjusted(self._dst_offset_ny)
            adjusted.append(w)
        return adjusted

    def auto_detect_dst(self, dt: Optional[datetime] = None) -> None:
        """Auto-detect DST offsets using zoneinfo (Python 3.9+ stdlib).
        In summer: London/NY sessions shift 1 hour earlier (offset = -60).
        """
        dt = dt or self.now_utc()
        try:
            from zoneinfo import ZoneInfo
            london = dt.astimezone(ZoneInfo("Europe/London"))
            ny = dt.astimezone(ZoneInfo("America/New_York"))
            # UTC offset in minutes; DST adds +60 to local time, so sessions
            # move -60 in UTC terms
            london_utcoff = london.utcoffset().total_seconds() / 60
            ny_utcoff = ny.utcoffset().total_seconds() / 60
            # London is UTC+0 in winter, UTC+1 in summer → session offset = -(utcoff)
            self._dst_offset_london = -int(london_utcoff)
            # NY is UTC-5 in winter, UTC-4 in summer → session offset = -(utcoff + 300)
            self._dst_offset_ny = -int(ny_utcoff + 300)
        except ImportError:
            pass  # zoneinfo not available; keep manual offsets

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
        """Minutes to current session close. Returns minutes to next session open if in DEAD_ZONE."""
        dt = dt or self.now_utc()
        session = self.current_session(dt)
        current_minutes = dt.hour * 60 + dt.minute

        if session == Session.DEAD_ZONE:
            # Return minutes to next session open
            best = float('inf')
            for window in SESSION_WINDOWS:
                open_min = window.open_hour * 60 + window.open_minute
                diff = open_min - current_minutes
                if diff < 0:
                    diff += 24 * 60
                if diff > 0:
                    best = min(best, diff)
            return best

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
        # Pre-build sorted high-impact times for O(log n) lockout check
        self._news_times = [
            e["time"] for e in self._news_calendar if e.get("impact") == "HIGH"
        ]

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
        """Check if in a news lockout window. Uses binary search for O(log n) performance."""
        dt = dt or self.now_utc()
        if not self._news_times:
            return False
        pre_mins = SENTINEL.pre_news_lockout_minutes
        post_mins = SENTINEL.post_news_lockout_minutes
        # Binary search: find nearest high-impact event
        idx = bisect.bisect_right(self._news_times, dt)
        # Check the event just before and just after the current time
        for i in (idx - 1, idx):
            if 0 <= i < len(self._news_times):
                event_time = self._news_times[i]
                pre_lockout = event_time - timedelta(minutes=pre_mins)
                post_lockout = event_time + timedelta(minutes=post_mins)
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
        """True if within last 2 business days of the month."""
        dt = dt or self.now_utc()
        next_month = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
        days_remaining = (next_month - dt).days
        # Account for weekends: if remaining days include only weekends, still month-end
        if days_remaining <= 2:
            return True
        # Check if remaining working days <= 2
        working_days = 0
        check = dt + timedelta(days=1)
        while check < next_month:
            if check.weekday() < 5:  # Mon-Fri
                working_days += 1
            check += timedelta(days=1)
        return working_days <= 2

    def is_quarter_end(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now_utc()
        return dt.month in (3, 6, 9, 12) and self.is_month_end(dt)

    def minutes_into_session(self, dt: Optional[datetime] = None) -> float:
        """Minutes elapsed since the current session opened. 0 if in DEAD_ZONE."""
        dt = dt or self.now_utc()
        session = self.current_session(dt)
        if session == Session.DEAD_ZONE:
            return 0.0
        current_minutes = dt.hour * 60 + dt.minute
        for window in self._get_adjusted_windows():
            if window.name == session:
                start = window.open_hour * 60 + window.open_minute
                return max(0.0, current_minutes - start)
        return 0.0

    def session_duration_minutes(self, dt: Optional[datetime] = None) -> float:
        """Duration of the current session in minutes. 0 if in DEAD_ZONE."""
        dt = dt or self.now_utc()
        session = self.current_session(dt)
        if session == Session.DEAD_ZONE:
            return 0.0
        for window in self._get_adjusted_windows():
            if window.name == session:
                start = window.open_hour * 60 + window.open_minute
                end = window.close_hour * 60 + window.close_minute
                return max(0.0, end - start)
        return 0.0

    def session_progress(self, dt: Optional[datetime] = None) -> float:
        """Fraction through current session [0.0, 1.0]. 0 if in DEAD_ZONE."""
        dur = self.session_duration_minutes(dt)
        if dur == 0:
            return 0.0
        return min(1.0, self.minutes_into_session(dt) / dur)

    def session_features(self, dt: Optional[datetime] = None) -> dict:
        """Return session-aware features including cyclical time encoding for ML models."""
        dt = dt or self.now_utc()
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.weekday()  # 0=Mon, 6=Sun
        day_of_month = dt.day

        # Cyclical encoding — maps time features onto unit circle for continuity
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)
        dom_sin = math.sin(2 * math.pi * day_of_month / 31)
        dom_cos = math.cos(2 * math.pi * day_of_month / 31)

        return {
            "session": self.current_session(dt).name,
            "minutes_to_london_open": self.minutes_to_session(Session.LONDON, dt),
            "minutes_to_ny_open": self.minutes_to_session(Session.NEW_YORK, dt),
            "minutes_to_session_close": self.minutes_to_close(dt),
            "minutes_into_session": self.minutes_into_session(dt),
            "session_duration_minutes": self.session_duration_minutes(dt),
            "session_progress": self.session_progress(dt),
            "day_of_week": self.day_of_week(dt),
            "week_of_month": self.week_of_month(dt),
            "minutes_to_next_news": self.minutes_to_next_news(dt),
            "last_news_minutes_ago": self.last_high_impact_news_minutes(dt),
            "is_month_end": self.is_month_end(dt),
            "is_quarter_end": self.is_quarter_end(dt),
            "is_friday_lockout": self.is_friday_lockout(dt),
            "is_news_lockout": self.is_news_lockout(dt),
            "market_open": self.is_market_open(dt),
            "is_trading_session": self.is_trading_session(dt),
            # Cyclical time encoding for ML
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "dom_sin": dom_sin,
            "dom_cos": dom_cos,
        }
