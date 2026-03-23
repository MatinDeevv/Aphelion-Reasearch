"""
APHELION Paper Trading Data Feed
Abstraction layer that provides bar-by-bar data from multiple sources:
  - LIVE: Real-time bars via MT5Connection (bar-level polling)
  - REPLAY: Replays historical Bar lists at real speed (or accelerated)
  - MT5_TICK: Production tick-level streaming with bar aggregation (Phase 5)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import AsyncIterator, Optional

import numpy as np

from aphelion.core.config import Timeframe, TIMEFRAMES, EventTopic
from aphelion.core.data_layer import Bar, Tick, BarAggregator
from aphelion.core.event_bus import EventBus, Event, Priority

logger = logging.getLogger(__name__)


class FeedMode(Enum):
    LIVE = auto()
    REPLAY = auto()
    MT5_TICK = auto()


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


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 TICK-LEVEL FEED — Phase 5  (Production-grade)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeedConfig:
    """Configuration for the MT5 tick-level data feed."""
    symbol: str = "XAUUSD"
    poll_interval_ms: int = 100                 # Tick poll interval (ms)
    reconnect_delay_s: float = 5.0              # Seconds between reconnection attempts
    max_reconnect_attempts: int = 10            # 0 = infinite
    bar_timeframes: list[Timeframe] = field(default_factory=lambda: [
        Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1,
    ])
    warmup_bars: int = 200                      # Bars to pre-load on connect
    warmup_timeframe: str = "1m"                # Timeframe string for warmup bars
    stale_tick_threshold_s: float = 30.0        # Alert if no tick for this long
    max_tick_buffer: int = 5000                 # Tick ring-buffer size


class FeedStats:
    """Runtime statistics for the MT5TickFeed."""
    __slots__ = (
        "ticks_received", "ticks_per_minute", "bars_emitted",
        "reconnect_count", "errors", "last_tick_time",
        "uptime_seconds", "_start_time", "_minute_tick_count",
        "_minute_start",
    )

    def __init__(self) -> None:
        self.ticks_received: int = 0
        self.ticks_per_minute: float = 0.0
        self.bars_emitted: int = 0
        self.reconnect_count: int = 0
        self.errors: int = 0
        self.last_tick_time: float = 0.0
        self.uptime_seconds: float = 0.0
        self._start_time: float = time.monotonic()
        self._minute_tick_count: int = 0
        self._minute_start: float = time.monotonic()

    def record_tick(self) -> None:
        self.ticks_received += 1
        self._minute_tick_count += 1
        self.last_tick_time = time.monotonic()
        now = time.monotonic()
        elapsed = now - self._minute_start
        if elapsed >= 60.0:
            self.ticks_per_minute = self._minute_tick_count * 60.0 / elapsed
            self._minute_tick_count = 0
            self._minute_start = now
        self.uptime_seconds = now - self._start_time

    def record_bar(self) -> None:
        self.bars_emitted += 1

    def record_error(self) -> None:
        self.errors += 1

    def record_reconnect(self) -> None:
        self.reconnect_count += 1

    def to_dict(self) -> dict:
        return {
            "ticks_received": self.ticks_received,
            "ticks_per_minute": round(self.ticks_per_minute, 1),
            "bars_emitted": self.bars_emitted,
            "reconnect_count": self.reconnect_count,
            "errors": self.errors,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }


class MT5TickFeed:
    """
    Production tick-level data feed from MetaTrader 5.

    Polls MT5Connection for ticks at configurable intervals, aggregates them
    into bars via BarAggregator, and publishes events via EventBus.

    Architecture:
        MT5Connection.get_last_tick() → dedup → validate → BarAggregator(s)
        → EventBus.publish(tick / bar.M1 / bar.M5 / bar.M15 / bar.H1)

    Lifecycle:
        feed = MT5TickFeed(mt5_conn, event_bus, config)
        await feed.start()     # Connects + warms up + begins polling
        ...                    # Events flow via EventBus
        await feed.stop()      # Graceful shutdown
    """

    def __init__(
        self,
        mt5_connection,               # MT5Connection (duck-typed)
        event_bus: EventBus,
        config: Optional[FeedConfig] = None,
    ):
        self._mt5 = mt5_connection
        self._bus = event_bus
        self._config = config or FeedConfig()

        # Bar aggregators — one per timeframe
        self._aggregators: dict[Timeframe, BarAggregator] = {
            tf: BarAggregator(tf) for tf in self._config.bar_timeframes
        }

        # Tick deduplication
        self._last_tick_ts: float = 0.0
        self._last_tick_bid: float = 0.0
        self._last_tick_ask: float = 0.0

        # Tick ring buffer for inspection / replay
        self._tick_buffer: deque[Tick] = deque(maxlen=self._config.max_tick_buffer)

        # Runtime
        self._running: bool = False
        self._connected: bool = False
        self._poll_task: Optional[asyncio.Task] = None
        self._stats = FeedStats()

        # Completed bars queue (for the bars() async iterator bridge)
        self._bar_queue: asyncio.Queue[Bar] = asyncio.Queue(maxsize=500)

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """
        Connect to MT5, fetch warmup bars, and begin tick polling.
        Returns True if successfully connected and started.
        """
        connected = await self._connect()
        if not connected:
            return False

        # Fetch warmup bars
        try:
            warmup = self.fetch_warmup_bars(
                self._config.warmup_timeframe,
                self._config.warmup_bars,
            )
            if warmup:
                for bar in warmup:
                    await self._bar_queue.put(bar)
                    self._stats.record_bar()
                logger.info("MT5TickFeed warmup complete — %d bars loaded", len(warmup))
                await self._bus.publish(Event(
                    topic=EventTopic.SYSTEM,
                    data={"type": "feed.warmup_complete", "bars": len(warmup)},
                    source="MT5TickFeed",
                ))
        except Exception as exc:
            logger.warning("Warmup failed: %s", exc)
            self._stats.record_error()

        # Start tick polling loop
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("MT5TickFeed started — polling every %dms", self._config.poll_interval_ms)

        await self._bus.publish(Event(
            topic=EventTopic.SYSTEM,
            data={"type": "feed.connected", "symbol": self._config.symbol},
            source="MT5TickFeed",
        ))

        return True

    async def stop(self) -> None:
        """Graceful shutdown: stop polling, disconnect."""
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._connected = False
        logger.info(
            "MT5TickFeed stopped — %d ticks, %d bars, %d errors",
            self._stats.ticks_received,
            self._stats.bars_emitted,
            self._stats.errors,
        )
        await self._bus.publish(Event(
            topic=EventTopic.SYSTEM,
            data={"type": "feed.disconnected", "stats": self._stats.to_dict()},
            source="MT5TickFeed",
        ))

    async def bars(self) -> AsyncIterator[Bar]:
        """
        Async iterator of completed bars.
        Bridges the tick-driven aggregation to PaperSession's bar-consuming loop.
        """
        while self._running or not self._bar_queue.empty():
            try:
                bar = await asyncio.wait_for(self._bar_queue.get(), timeout=1.0)
                yield bar
            except asyncio.TimeoutError:
                continue

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_stats(self) -> FeedStats:
        return self._stats

    def fetch_warmup_bars(self, timeframe_str: str = "1m", count: int = 200) -> list[Bar]:
        """
        Fetch historical bars from MT5 for warmup.
        Adds missing timeframe/spread fields to bars if needed.
        """
        raw_bars = self._mt5.get_bars(timeframe_str, count)
        tf_map = {"1m": Timeframe.M1, "5m": Timeframe.M5, "15m": Timeframe.M15, "1h": Timeframe.H1}
        target_tf = tf_map.get(timeframe_str, Timeframe.M1)

        result: list[Bar] = []
        for bar in raw_bars:
            # Ensure timeframe and spread are set (MT5Connection may omit them)
            if not hasattr(bar, "timeframe") or bar.timeframe is None:
                bar = Bar(
                    timestamp=bar.timestamp,
                    timeframe=target_tf,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    tick_volume=getattr(bar, "tick_volume", 0),
                    spread=getattr(bar, "spread", 0.0),
                    is_complete=True,
                )
            result.append(bar)
        return result

    # ── Internal: connection ──────────────────────────────────────────────

    async def _connect(self) -> bool:
        """Connect to MT5 with retry logic."""
        max_attempts = self._config.max_reconnect_attempts or 999
        for attempt in range(1, max_attempts + 1):
            try:
                ok = self._mt5.connect()
                if ok:
                    self._connected = True
                    logger.info("MT5TickFeed connected (attempt %d)", attempt)
                    return True
            except Exception as exc:
                logger.warning("MT5 connect attempt %d failed: %s", attempt, exc)
                self._stats.record_error()

            if attempt < max_attempts:
                await asyncio.sleep(self._config.reconnect_delay_s)

        logger.error("MT5TickFeed failed to connect after %d attempts", max_attempts)
        self._connected = False
        return False

    async def _reconnect(self) -> bool:
        """Reconnect after a connection drop."""
        self._connected = False
        self._stats.record_reconnect()
        logger.warning("MT5TickFeed reconnecting...")

        await self._bus.publish(Event(
            topic=EventTopic.SYSTEM,
            data={"type": "feed.reconnecting"},
            source="MT5TickFeed",
        ))

        ok = await self._connect()

        if ok:
            await self._bus.publish(Event(
                topic=EventTopic.SYSTEM,
                data={"type": "feed.reconnected"},
                source="MT5TickFeed",
            ))

        return ok

    # ── Internal: poll loop ───────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main tick polling loop. Runs until stop() is called."""
        poll_s = self._config.poll_interval_ms / 1000.0
        consecutive_errors = 0
        stale_warned = False

        while self._running:
            try:
                tick = self._mt5.get_last_tick()
                if tick is None:
                    # Check for stale data
                    if self._stats.last_tick_time > 0:
                        elapsed = time.monotonic() - self._stats.last_tick_time
                        if elapsed > self._config.stale_tick_threshold_s and not stale_warned:
                            stale_warned = True
                            logger.warning("Stale tick data: %.1fs since last tick", elapsed)
                            await self._bus.publish(Event(
                                topic=EventTopic.SYSTEM,
                                data={"type": "feed.stale", "seconds": elapsed},
                                source="MT5TickFeed",
                                priority=Priority.HIGH,
                            ))
                    await asyncio.sleep(poll_s)
                    continue

                stale_warned = False
                consecutive_errors = 0

                # Deduplication: skip if same timestamp + bid + ask
                tick_ts = tick.timestamp
                if isinstance(tick_ts, datetime):
                    tick_ts = tick_ts.timestamp()
                if (tick_ts == self._last_tick_ts
                        and tick.bid == self._last_tick_bid
                        and tick.ask == self._last_tick_ask):
                    await asyncio.sleep(poll_s)
                    continue

                self._last_tick_ts = tick_ts
                self._last_tick_bid = tick.bid
                self._last_tick_ask = tick.ask

                # Normalize tick timestamp to float for BarAggregator
                if isinstance(tick.timestamp, datetime):
                    tick = Tick(
                        timestamp=tick.timestamp.timestamp(),
                        bid=tick.bid,
                        ask=tick.ask,
                        last=tick.last,
                        volume=tick.volume,
                        flags=getattr(tick, "flags", 0),
                    )

                self._tick_buffer.append(tick)
                self._stats.record_tick()

                # Publish tick event
                await self._bus.publish(Event(
                    topic=EventTopic.TICK,
                    data=tick,
                    source="MT5TickFeed",
                ))

                # Feed tick to all bar aggregators
                for tf, agg in self._aggregators.items():
                    completed_bar = agg.process_tick(tick)
                    if completed_bar is not None:
                        self._stats.record_bar()
                        await self._bar_queue.put(completed_bar)
                        # Publish bar event per timeframe
                        await self._bus.publish(Event(
                            topic=EventTopic.BAR,
                            data={"timeframe": tf.value, "bar": completed_bar},
                            source="MT5TickFeed",
                        ))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_errors += 1
                self._stats.record_error()
                logger.exception("MT5TickFeed poll error (%d consecutive)", consecutive_errors)

                if consecutive_errors >= 5:
                    # Likely disconnected — attempt reconnect
                    ok = await self._reconnect()
                    if not ok:
                        logger.error("MT5TickFeed giving up after reconnect failed")
                        self._running = False
                        await self._bus.publish(Event(
                            topic=EventTopic.SYSTEM,
                            data={"type": "feed.error", "fatal": True, "msg": str(exc)},
                            source="MT5TickFeed",
                            priority=Priority.CRITICAL,
                        ))
                        break
                    consecutive_errors = 0

            await asyncio.sleep(poll_s)
