"""
ATLAS LIVE — Real-Time Macro Intelligence Feed
Phase 19 — Engineering Spec v3.0

Provides live macro context to ARES without blocking the M1 loop.
All feeds run async; stale data is marked and ARES ignores it.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from enum import Enum

import json
import os


class DataFreshness(Enum):
    FRESH = "FRESH"       # < 1 hour old
    STALE = "STALE"       # 1-4 hours old
    EXPIRED = "EXPIRED"   # > 4 hours old


@dataclass
class FeedData:
    """A single data feed with freshness tracking."""
    name: str
    value: float
    direction: int = 0          # 1=rising, -1=falling, 0=flat
    confidence: float = 0.0
    timestamp: Optional[datetime] = None
    freshness: DataFreshness = DataFreshness.EXPIRED

    def update_freshness(self, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        if self.timestamp is None:
            self.freshness = DataFreshness.EXPIRED
            return
        age = now - self.timestamp
        if age < timedelta(hours=1):
            self.freshness = DataFreshness.FRESH
        elif age < timedelta(hours=4):
            self.freshness = DataFreshness.STALE
        else:
            self.freshness = DataFreshness.EXPIRED


@dataclass
class DXYFeed:
    """US Dollar Index feed."""
    value: float = 0.0
    sma_20: float = 0.0
    trend: int = 0           # 1=strengthening, -1=weakening
    correlation_with_gold: float = -0.5
    timestamp: Optional[datetime] = None

    def compute_gold_bias(self) -> int:
        """DXY strengthening => gold bearish; weakening => gold bullish."""
        if self.trend == 1 and self.correlation_with_gold < -0.3:
            return -1   # Bearish for gold
        elif self.trend == -1 and self.correlation_with_gold < -0.3:
            return 1    # Bullish for gold
        return 0


@dataclass
class COTData:
    """Commitment of Traders data for gold futures."""
    report_date: Optional[datetime] = None
    managed_money_long: int = 0
    managed_money_short: int = 0
    commercial_long: int = 0
    commercial_short: int = 0

    @property
    def net_speculative(self) -> int:
        return self.managed_money_long - self.managed_money_short

    @property
    def net_commercial(self) -> int:
        return self.commercial_long - self.commercial_short

    def speculative_bias(self) -> int:
        """Positive = bullish positioning, negative = bearish."""
        net = self.net_speculative
        if net > 100_000:
            return 1
        elif net < -50_000:
            return -1
        return 0


class FedCalendar:
    """Federal Reserve event calendar."""

    FOMC_DATES_2025 = [
        "2025-01-29", "2025-03-19", "2025-05-07",
        "2025-06-18", "2025-07-30", "2025-09-17",
        "2025-10-29", "2025-12-10",
    ]

    NFP_DATES_2025 = [
        "2025-01-10", "2025-02-07", "2025-03-07",
        "2025-04-04", "2025-05-02", "2025-06-06",
        "2025-07-03", "2025-08-01", "2025-09-05",
        "2025-10-03", "2025-11-07", "2025-12-05",
    ]

    CPI_DATES_2025 = [
        "2025-01-15", "2025-02-12", "2025-03-12",
        "2025-04-10", "2025-05-13", "2025-06-11",
        "2025-07-11", "2025-08-12", "2025-09-10",
        "2025-10-14", "2025-11-12", "2025-12-10",
    ]

    def __init__(self):
        self._events: List[dict] = []
        # Pre-load known events
        for date_str in self.FOMC_DATES_2025:
            self._events.append({"date": date_str, "type": "FOMC", "impact": "HIGH"})
        for date_str in self.NFP_DATES_2025:
            self._events.append({"date": date_str, "type": "NFP", "impact": "HIGH"})
        for date_str in self.CPI_DATES_2025:
            self._events.append({"date": date_str, "type": "CPI", "impact": "HIGH"})

    def is_near_event(self, dt: datetime, hours_before: int = 2, hours_after: int = 4) -> bool:
        """Check if we're within a dangerous event window."""
        date_str = dt.strftime("%Y-%m-%d")
        for event in self._events:
            if event["date"] == date_str:
                return True
        return False

    def next_event(self, dt: datetime) -> Optional[dict]:
        """Get the next upcoming event."""
        date_str = dt.strftime("%Y-%m-%d")
        for event in sorted(self._events, key=lambda e: e["date"]):
            if event["date"] >= date_str:
                return event
        return None


@dataclass
class AtlasState:
    """Complete ATLAS LIVE state for ARES consumption."""
    dxy_bias: int = 0
    cot_bias: int = 0
    near_fed_event: bool = False
    seasonal_bias: float = 0.0
    safe_to_trade: bool = True
    feeds_fresh: int = 0
    feeds_total: int = 0
    timestamp: Optional[datetime] = None


class AtlasLive:
    """
    Master ATLAS LIVE coordinator.
    Aggregates DXY, COT, Fed calendar, seasonality into a single state object.
    """

    def __init__(self):
        self._dxy = DXYFeed()
        self._cot = COTData()
        self._fed = FedCalendar()
        self._feeds: Dict[str, FeedData] = {}
        self._state = AtlasState()

    def update_dxy(self, value: float, sma_20: float, correlation: float = -0.5) -> None:
        self._dxy.value = value
        self._dxy.sma_20 = sma_20
        self._dxy.correlation_with_gold = correlation
        self._dxy.trend = 1 if value > sma_20 else -1 if value < sma_20 else 0
        self._dxy.timestamp = datetime.now(timezone.utc)

    def update_cot(
        self, mm_long: int, mm_short: int,
        comm_long: int, comm_short: int,
    ) -> None:
        self._cot.managed_money_long = mm_long
        self._cot.managed_money_short = mm_short
        self._cot.commercial_long = comm_long
        self._cot.commercial_short = comm_short
        self._cot.report_date = datetime.now(timezone.utc)

    def update_feed(self, name: str, value: float, direction: int = 0) -> None:
        self._feeds[name] = FeedData(
            name=name, value=value, direction=direction,
            timestamp=datetime.now(timezone.utc),
            freshness=DataFreshness.FRESH,
        )

    def compute_state(self, now: Optional[datetime] = None) -> AtlasState:
        """Compute the aggregated ATLAS state for ARES."""
        now = now or datetime.now(timezone.utc)

        # Update feed freshness
        fresh_count = 0
        for feed in self._feeds.values():
            feed.update_freshness(now)
            if feed.freshness == DataFreshness.FRESH:
                fresh_count += 1

        self._state = AtlasState(
            dxy_bias=self._dxy.compute_gold_bias(),
            cot_bias=self._cot.speculative_bias(),
            near_fed_event=self._fed.is_near_event(now),
            safe_to_trade=not self._fed.is_near_event(now),
            feeds_fresh=fresh_count,
            feeds_total=len(self._feeds),
            timestamp=now,
        )
        return self._state

    @property
    def dxy(self) -> DXYFeed:
        return self._dxy

    @property
    def cot(self) -> COTData:
        return self._cot

    @property
    def state(self) -> AtlasState:
        return self._state
