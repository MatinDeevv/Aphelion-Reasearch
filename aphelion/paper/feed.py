"""
APHELION Paper Trading Data Feed
Abstraction layer that provides bar-by-bar data from multiple sources:
  - LIVE: Real-time bars via MT5Connection
  - REPLAY: Replays historical Bar lists at real speed (or accelerated)
  - SIMULATED: Random-walk price generator for testing
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import AsyncIterator, Optional

import numpy as np

from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar

logger = logging.getLogger(__name__)


class FeedMode(Enum):
    LIVE = auto()
    REPLAY = auto()
    SIMULATED = auto()


# ── Abstract base ────────────────────────────────────────────────────────────

class DataFeed(ABC):
    """Abstract async bar feed for paper trading."""

    @abstractmethod
    async def bars(self) -> AsyncIterator[Bar]:
        """Yield bars one at a time."""
        ...  # pragma: no cover

    @abstractmethod
    def stop(self) -> None:
        """Signal the feed to stop yielding."""
        ...  # pragma: no cover


# ── Live MT5 feed ────────────────────────────────────────────────────────────

class LiveMT5Feed(DataFeed):
    """
    Polls MT5 for completed bars at a fixed interval.
    Uses MT5Connection.get_bars() to fetch the latest bar
    and yields only NEW bars (deduplication by timestamp).
    """

    def __init__(
        self,
        mt5_connection,  # MT5Connection instance (duck-typed to avoid circular import)
        timeframe: str = "1m",
        poll_interval_seconds: float = 5.0,
    ):
        self._mt5 = mt5_connection
        self._timeframe = timeframe
        self._poll_interval = poll_interval_seconds
        self._running = True
        self._last_bar_time: Optional[datetime] = None

    async def bars(self) -> AsyncIterator[Bar]:
        """Yield new bars as they appear on MT5."""
        logger.info("LiveMT5Feed started — polling every %.1fs", self._poll_interval)
        while self._running:
            try:
                recent = self._mt5.get_bars(self._timeframe, count=5)
                if recent:
                    for bar in recent:
                        bar_ts = bar.timestamp if isinstance(bar.timestamp, datetime) else None
                        if bar_ts and (self._last_bar_time is None or bar_ts > self._last_bar_time):
                            self._last_bar_time = bar_ts
                            yield bar
            except Exception:
                logger.exception("LiveMT5Feed error fetching bars")

            await asyncio.sleep(self._poll_interval)

    def stop(self) -> None:
        self._running = False


# ── Historical replay feed ───────────────────────────────────────────────────

class ReplayFeed(DataFeed):
    """
    Replays a pre-loaded list of bars.
    Optionally sleeps between bars to simulate real-time pacing.
    """

    def __init__(
        self,
        bar_list: list[Bar],
        speed_multiplier: float = 1.0,
        realtime_pacing: bool = False,
    ):
        self._bars = bar_list
        self._speed = max(0.01, speed_multiplier)
        self._pacing = realtime_pacing
        self._running = True

    async def bars(self) -> AsyncIterator[Bar]:
        """Yield bars from the list, optionally paced."""
        logger.info("ReplayFeed started — %d bars, speed=%.1fx", len(self._bars), self._speed)
        prev_ts: Optional[datetime] = None

        for bar in self._bars:
            if not self._running:
                break

            if self._pacing and prev_ts is not None:
                bar_ts = bar.timestamp if isinstance(bar.timestamp, datetime) else None
                if bar_ts:
                    delta = (bar_ts - prev_ts).total_seconds()
                    sleep_time = max(0, delta / self._speed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

            prev_ts = bar.timestamp if isinstance(bar.timestamp, datetime) else prev_ts
            yield bar

        logger.info("ReplayFeed exhausted")

    def stop(self) -> None:
        self._running = False


# ── Simulated random-walk feed ───────────────────────────────────────────────

@dataclass
class SimulatedFeedConfig:
    """Configuration for the random-walk test feed."""
    start_price: float = 2350.0
    volatility: float = 0.001          # Per-bar return std dev
    trend: float = 0.0                 # Per-bar drift
    bar_interval_seconds: float = 60   # Time between bars
    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.M1
    seed: int = 42
    max_bars: int = 0                  # 0 = infinite


class SimulatedFeed(DataFeed):
    """
    Generates synthetic bars via geometric random walk.
    Useful for testing the paper session without MT5.
    """

    def __init__(self, config: Optional[SimulatedFeedConfig] = None):
        self._config = config or SimulatedFeedConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._running = True

    async def bars(self) -> AsyncIterator[Bar]:
        """Yield synthetic bars continuously."""
        price = self._config.start_price
        bar_count = 0
        ts = datetime.now(timezone.utc)
        interval = timedelta(seconds=self._config.bar_interval_seconds)

        logger.info(
            "SimulatedFeed started — price=%.2f, vol=%.4f",
            price, self._config.volatility,
        )

        while self._running:
            if 0 < self._config.max_bars <= bar_count:
                break

            # Generate OHLCV via random walk
            returns = self._rng.normal(self._config.trend, self._config.volatility, 4)
            o = price
            h = o * (1 + abs(returns[0]))
            l = o * (1 - abs(returns[1]))
            c = o * (1 + returns[2])
            v = max(100, self._rng.poisson(500))

            # Ensure OHLC consistency
            h = max(o, h, c)
            l = min(o, l, c)

            bar = Bar(
                timestamp=ts,
                timeframe=self._config.timeframe,
                open=round(o, 2),
                high=round(h, 2),
                low=round(l, 2),
                close=round(c, 2),
                volume=float(v),
                tick_volume=int(v),
                spread=round(abs(returns[3]) * o * 0.01, 2),
                is_complete=True,
            )

            yield bar

            price = c
            ts += interval
            bar_count += 1

            # Small async yield to keep the event loop responsive
            await asyncio.sleep(0)

        logger.info("SimulatedFeed stopped after %d bars", bar_count)

    def stop(self) -> None:
        self._running = False
