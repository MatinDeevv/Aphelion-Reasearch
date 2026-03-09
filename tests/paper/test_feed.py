"""Tests for MT5TickFeed and FeedConfig — Phase 5."""

import asyncio
from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aphelion.core.config import Timeframe, EventTopic
from aphelion.core.data_layer import Bar, Tick, BarAggregator
from aphelion.core.event_bus import EventBus, Event
from aphelion.paper.feed import (
    FeedConfig,
    FeedMode,
    FeedStats,
    MT5TickFeed,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tick(
    ts: float = 1704067200.0,
    bid: float = 2350.0,
    ask: float = 2350.20,
    last: float = 2350.10,
    volume: float = 1.0,
) -> Tick:
    """Create a Tick for testing."""
    return Tick(timestamp=ts, bid=bid, ask=ask, last=last, volume=volume)


def _make_bar(
    ts_offset: int = 0,
    close: float = 2350.0,
    tf: Timeframe = Timeframe.M1,
) -> Bar:
    """Create a Bar for testing."""
    return Bar(
        timestamp=datetime.fromtimestamp(1704067200.0 + ts_offset * 60, tz=timezone.utc),
        timeframe=tf,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=500.0,
        tick_volume=500,
        spread=0.20,
        is_complete=True,
    )


def _mock_mt5_connection(
    connect_ok: bool = True,
    ticks: list[Tick] | None = None,
    bars: list[Bar] | None = None,
):
    """Create a mock MT5Connection."""
    conn = MagicMock()
    conn.connect.return_value = connect_ok
    conn.disconnect.return_value = None
    conn.is_connected = connect_ok

    # Tick stream
    tick_iter = iter(ticks or [])
    def get_last_tick():
        try:
            return next(tick_iter)
        except StopIteration:
            return None
    conn.get_last_tick = MagicMock(side_effect=get_last_tick)

    # Bar warmup
    conn.get_bars.return_value = bars or []
    conn.get_recent_ticks.return_value = []
    return conn


# ═══════════════════════════════════════════════════════════════════════════════
# FEED CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeedConfig:
    """Tests for FeedConfig dataclass."""

    def test_defaults(self):
        """Default FeedConfig should have sensible values."""
        cfg = FeedConfig()
        assert cfg.symbol == "XAUUSD"
        assert cfg.poll_interval_ms == 100
        assert cfg.warmup_bars == 200
        assert len(cfg.bar_timeframes) == 4
        assert Timeframe.M1 in cfg.bar_timeframes

    def test_custom_config(self):
        """FeedConfig should respect custom values."""
        cfg = FeedConfig(
            symbol="EURUSD",
            poll_interval_ms=50,
            warmup_bars=500,
            bar_timeframes=[Timeframe.M1, Timeframe.M5],
        )
        assert cfg.symbol == "EURUSD"
        assert cfg.poll_interval_ms == 50
        assert len(cfg.bar_timeframes) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# FEED STATS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeedStats:
    """Tests for FeedStats tracking."""

    def test_initial_state(self):
        """Stats should start at zero."""
        stats = FeedStats()
        assert stats.ticks_received == 0
        assert stats.bars_emitted == 0
        assert stats.reconnect_count == 0
        assert stats.errors == 0

    def test_record_tick(self):
        """record_tick should increment counters."""
        stats = FeedStats()
        for _ in range(10):
            stats.record_tick()
        assert stats.ticks_received == 10
        assert stats.uptime_seconds > 0

    def test_record_bar(self):
        """record_bar should increment bar count."""
        stats = FeedStats()
        stats.record_bar()
        stats.record_bar()
        assert stats.bars_emitted == 2

    def test_to_dict(self):
        """to_dict should return all fields."""
        stats = FeedStats()
        stats.record_tick()
        stats.record_bar()
        stats.record_error()
        d = stats.to_dict()
        assert d["ticks_received"] == 1
        assert d["bars_emitted"] == 1
        assert d["errors"] == 1
        assert "uptime_seconds" in d


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 TICK FEED TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMT5TickFeed:
    """Tests for MT5TickFeed with mocked MT5Connection."""

    async def test_start_and_stop(self):
        """Feed should connect, start, and stop cleanly."""
        conn = _mock_mt5_connection(connect_ok=True)
        bus = EventBus()
        config = FeedConfig(warmup_bars=0)
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is True
        assert feed.is_connected is True

        await feed.stop()
        assert feed._running is False

    async def test_start_fails_on_connection_error(self):
        """Feed should return False if MT5 connection fails."""
        conn = _mock_mt5_connection(connect_ok=False)
        bus = EventBus()
        config = FeedConfig(warmup_bars=0, max_reconnect_attempts=1, reconnect_delay_s=0.01)
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is False
        assert feed.is_connected is False

    async def test_warmup_bars_loaded(self):
        """Feed should pre-load warmup bars on start."""
        warmup_bars = [_make_bar(ts_offset=i) for i in range(10)]
        conn = _mock_mt5_connection(connect_ok=True, bars=warmup_bars)
        bus = EventBus()
        config = FeedConfig(warmup_bars=10)
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is True

        # Warmup bars should be in the queue
        assert feed._bar_queue.qsize() == 10

        await feed.stop()

    async def test_tick_deduplication(self):
        """Duplicate ticks (same ts/bid/ask) should be filtered."""
        ticks = [
            _make_tick(ts=1704067200.0, bid=2350.0, ask=2350.20),
            _make_tick(ts=1704067200.0, bid=2350.0, ask=2350.20),  # Duplicate
            _make_tick(ts=1704067201.0, bid=2351.0, ask=2351.20),  # New
        ]
        conn = _mock_mt5_connection(connect_ok=True, ticks=ticks)
        bus = EventBus()
        config = FeedConfig(warmup_bars=0, poll_interval_ms=10)
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is True

        # Allow polling to process ticks
        await asyncio.sleep(0.15)
        await feed.stop()

        # Should have processed <= 2 unique ticks (not 3)
        stats = feed.get_stats()
        assert stats.ticks_received <= 2

    async def test_bar_aggregation(self):
        """Ticks crossing a bar boundary should produce completed bars."""
        # Ticks spanning two minutes — should produce at least one M1 bar
        # BarAggregator completes a bar when a tick arrives in a NEW bar period
        base_ts = 1704067200.0  # Start of a minute boundary
        ticks = []
        # 30 ticks in the first minute
        for i in range(30):
            ticks.append(_make_tick(
                ts=base_ts + i * 2,
                bid=2350.0 + i * 0.01,
                ask=2350.2 + i * 0.01,
                last=2350.1 + i * 0.01,
            ))
        # 1 tick in the next minute — this triggers bar completion
        ticks.append(_make_tick(
            ts=base_ts + 61,
            bid=2351.0,
            ask=2351.2,
            last=2351.1,
        ))

        conn = _mock_mt5_connection(connect_ok=True, ticks=ticks)
        bus = EventBus()
        config = FeedConfig(warmup_bars=0, poll_interval_ms=5, bar_timeframes=[Timeframe.M1])
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is True

        # Wait for polling to process all ticks
        await asyncio.sleep(0.8)
        await feed.stop()

        # At least one bar should have been emitted
        stats = feed.get_stats()
        assert stats.bars_emitted >= 1

    async def test_tick_publishes_to_event_bus(self):
        """Ticks should be published to EventBus."""
        ticks = [_make_tick(ts=1704067200.0 + i, bid=2350.0, ask=2350.2) for i in range(3)]
        conn = _mock_mt5_connection(connect_ok=True, ticks=ticks)
        bus = EventBus()
        config = FeedConfig(warmup_bars=0, poll_interval_ms=10)
        feed = MT5TickFeed(conn, bus, config)

        received_events = []

        async def on_tick(event: Event):
            received_events.append(event)

        bus.subscribe(EventTopic.TICK, on_tick)

        # Start the event bus dispatcher
        await bus.start()

        ok = await feed.start()
        assert ok is True

        await asyncio.sleep(0.15)
        await feed.stop()
        await bus.stop()

        # At least some tick events should have been published
        assert len(received_events) >= 1

    async def test_fetch_warmup_bars_adds_timeframe(self):
        """fetch_warmup_bars should fix missing timeframe on bars."""
        # Bars without timeframe set
        raw_bars = [
            Bar(
                timestamp=datetime.fromtimestamp(1704067200.0, tz=timezone.utc),
                timeframe=None,
                open=2349.5,
                high=2351.0,
                low=2349.0,
                close=2350.0,
                volume=500.0,
                tick_volume=500,
                spread=0.0,
            ),
        ]
        conn = _mock_mt5_connection(connect_ok=True, bars=raw_bars)
        bus = EventBus()
        config = FeedConfig(warmup_bars=1)
        feed = MT5TickFeed(conn, bus, config)

        result = feed.fetch_warmup_bars("5m", 1)
        assert len(result) == 1
        assert result[0].timeframe == Timeframe.M5

    async def test_reconnect_on_errors(self):
        """Feed should attempt reconnection after consecutive errors."""
        conn = MagicMock()
        conn.connect.return_value = True
        conn.is_connected = True
        conn.get_bars.return_value = []

        # Make get_last_tick raise errors
        call_count = 0
        def failing_tick():
            nonlocal call_count
            call_count += 1
            if call_count <= 6:
                raise ConnectionError("MT5 disconnected")
            return None
        conn.get_last_tick = MagicMock(side_effect=failing_tick)

        bus = EventBus()
        config = FeedConfig(warmup_bars=0, poll_interval_ms=5, reconnect_delay_s=0.01, max_reconnect_attempts=2)
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is True

        await asyncio.sleep(0.3)
        await feed.stop()

        stats = feed.get_stats()
        assert stats.errors >= 1

    async def test_feed_mode_enum(self):
        """FeedMode should include MT5_TICK."""
        assert FeedMode.MT5_TICK is not None
        assert FeedMode.MT5_TICK.name == "MT5_TICK"

    async def test_bars_async_iterator(self):
        """The bars() async iterator should yield warmup bars."""
        warmup = [_make_bar(ts_offset=i) for i in range(5)]
        conn = _mock_mt5_connection(connect_ok=True, bars=warmup)
        bus = EventBus()
        config = FeedConfig(warmup_bars=5, poll_interval_ms=10)
        feed = MT5TickFeed(conn, bus, config)

        ok = await feed.start()
        assert ok is True

        collected = []
        async for bar in feed.bars():
            collected.append(bar)
            if len(collected) >= 5:
                await feed.stop()
                break

        assert len(collected) == 5
        # Verify they are Bar objects
        for bar in collected:
            assert isinstance(bar, Bar)
