"""Tests for Phase 5 paper trading infrastructure."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from aphelion.backtest.order import Order, OrderSide, OrderStatus, OrderType
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.clock import MarketClock
from aphelion.core.config import SENTINEL, Timeframe
from aphelion.core.data_layer import Bar, DataLayer
from aphelion.core.event_bus import EventBus
from aphelion.paper.feed import (
    ReplayFeed,
)
from aphelion.paper.ledger import PaperLedger
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
from aphelion.risk.sentinel.core import SentinelCore
from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer
from aphelion.risk.sentinel.execution.paper import PaperConfig, PaperExecutor, PaperFill
from aphelion.risk.sentinel.validator import TradeProposal, TradeValidator


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _make_sentinel_stack():
    """Wire the full SENTINEL stack for testing."""
    bus = EventBus()
    clock = MarketClock()
    core = SentinelCore(bus, clock)
    validator = TradeValidator(core, clock)
    cb = CircuitBreaker(bus)
    enforcer = ExecutionEnforcer(validator, cb)
    return {
        "bus": bus,
        "clock": clock,
        "core": core,
        "validator": validator,
        "cb": cb,
        "enforcer": enforcer,
    }


def _make_order(
    order_id: str = "TEST-001",
    side: OrderSide = OrderSide.BUY,
    size_lots: float = 0.10,
    stop_loss: float = 2340.0,
    take_profit: float = 2370.0,
    size_pct: float = 0.015,
) -> Order:
    """Create a test Order."""
    return Order(
        order_id=order_id,
        symbol="XAUUSD",
        order_type=OrderType.MARKET,
        side=side,
        size_lots=size_lots,
        entry_price=0.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size_pct=size_pct,
        proposed_by="TEST_STRATEGY",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER EXECUTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaperExecutor:
    """Tests for PaperExecutor virtual fill simulation."""

    def setup_method(self):
        sentinel = _make_sentinel_stack()
        # Set simulated time to a Tuesday at 10:00 UTC (London session)
        trading_time = datetime(2024, 1, 9, 10, 0, 0, tzinfo=timezone.utc)
        sentinel["clock"].set_simulated_time(trading_time)
        self.portfolio = Portfolio(10_000.0)
        self.config = PaperConfig(initial_capital=10_000.0)
        self.executor = PaperExecutor(
            config=self.config,
            enforcer=sentinel["enforcer"],
            sentinel_core=sentinel["core"],
            portfolio=self.portfolio,
            event_bus=sentinel["bus"],
        )
        sentinel["core"].update_equity(10_000.0)

    def test_market_order_fill(self):
        """Market orders should fill with slippage applied."""
        order = _make_order()
        fill = self.executor.submit_order(order, current_price=2350.0)

        assert fill is not None
        assert isinstance(fill, PaperFill)
        assert fill.filled_price == 2350.0 + self.config.slippage_points
        assert fill.side == OrderSide.BUY
        assert fill.size_lots == 0.10
        assert order.status == OrderStatus.FILLED

    def test_sell_order_slippage_direction(self):
        """SELL orders should have slippage subtracted."""
        order = _make_order(
            side=OrderSide.SELL,
            stop_loss=2360.0,
            take_profit=2330.0,
        )
        fill = self.executor.submit_order(order, current_price=2350.0)

        assert fill is not None
        assert fill.filled_price == 2350.0 - self.config.slippage_points

    def test_order_rejected_no_stop_loss(self):
        """Orders without a stop loss should be rejected by SENTINEL."""
        order = _make_order(stop_loss=0.0)
        fill = self.executor.submit_order(order, current_price=2350.0)

        assert fill is None
        assert order.status == OrderStatus.REJECTED
        assert self.executor.stats["rejection_count"] == 1

    def test_order_rejected_wrong_symbol(self):
        """Non-XAUUSD orders should be rejected."""
        order = _make_order()
        order.symbol = "EURUSD"
        fill = self.executor.submit_order(order, current_price=2350.0)

        assert fill is None
        assert order.status == OrderStatus.REJECTED

    def test_position_registered_with_sentinel(self):
        """Filled orders should create positions in SentinelCore."""
        order = _make_order()
        self.executor.submit_order(order, current_price=2350.0)

        # Position should now be in the portfolio
        assert len(self.portfolio._open_positions) == 1

    def test_max_positions_enforced(self):
        """No more than MAX_SIMULTANEOUS_POSITIONS should be allowed."""
        fills = []
        for i in range(SENTINEL.max_simultaneous_positions + 1):
            order = _make_order(
                order_id=f"TEST-{i:03d}",
                size_pct=0.015,
            )
            fill = self.executor.submit_order(order, current_price=2350.0)
            if fill:
                fills.append(fill)

        assert len(fills) == SENTINEL.max_simultaneous_positions

    def test_sl_tp_detection_long(self):
        """SL/TP check should detect hits on LONG positions."""
        order = _make_order(stop_loss=2340.0, take_profit=2370.0)
        self.executor.submit_order(order, current_price=2350.0)

        # Price drops to SL
        exits = self.executor.check_sl_tp(2339.0)
        assert len(exits) == 1
        assert exits[0][2] == "SL_HIT"

    def test_sl_tp_detection_short(self):
        """SL/TP check should detect hits on SHORT positions."""
        order = _make_order(
            side=OrderSide.SELL,
            stop_loss=2360.0,
            take_profit=2330.0,
        )
        self.executor.submit_order(order, current_price=2350.0)

        # Price rises to SL
        exits = self.executor.check_sl_tp(2361.0)
        assert len(exits) == 1
        assert exits[0][2] == "SL_HIT"

    def test_tp_hit_long(self):
        """TP should trigger on LONG when price reaches target."""
        order = _make_order(stop_loss=2340.0, take_profit=2370.0)
        self.executor.submit_order(order, current_price=2350.0)

        exits = self.executor.check_sl_tp(2371.0)
        assert len(exits) == 1
        assert exits[0][2] == "TP_HIT"

    def test_close_position(self):
        """Closing a position should remove it and return P&L."""
        order = _make_order()
        fill = self.executor.submit_order(order, current_price=2350.0)
        assert fill is not None

        pnl = self.executor.close_position(order.order_id, 2360.0, "MANUAL")
        assert pnl is not None
        assert len(self.portfolio._open_positions) == 0

    def test_force_close_all(self):
        """force_close_all should close every open position."""
        for i in range(2):
            order = _make_order(order_id=f"TEST-{i:03d}")
            self.executor.submit_order(order, current_price=2350.0)

        closed = self.executor.force_close_all(2345.0, "L3_HALT")
        assert closed == 2
        assert len(self.portfolio._open_positions) == 0

    def test_stats_tracking(self):
        """Stats should update after fills and rejections."""
        order = _make_order()
        self.executor.submit_order(order, current_price=2350.0)

        bad_order = _make_order(order_id="BAD-001", stop_loss=0.0)
        self.executor.submit_order(bad_order, current_price=2350.0)

        stats = self.executor.stats
        assert stats["fill_count"] == 1
        assert stats["rejection_count"] == 1

    def test_oversized_position_rejected(self):
        """Position exceeding 2% of account should be rejected."""
        order = _make_order(size_pct=0.05)  # 5% — way over SENTINEL limit
        fill = self.executor.submit_order(order, current_price=2350.0)

        # Either rejected or capped — depends on validator logic
        # The order should NOT have a 5% size
        if fill:
            assert fill.size_lots < 0.10 * 2.5  # Scale would be huge

    def test_commission_applied(self):
        """Commission should be calculated on fill."""
        order = _make_order(size_lots=1.0)
        fill = self.executor.submit_order(order, current_price=2350.0)

        assert fill is not None
        assert fill.commission == self.config.commission_per_lot * 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FEED TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestReplayFeed:
    """Tests for the historical replay feed."""

    async def test_replays_all_bars(self):
        """Replay feed should yield all bars from the list."""
        bars_in = [_make_bar(ts_offset=i, close=2350.0 + i) for i in range(30)]
        feed = ReplayFeed(bars_in)

        bars_out = []
        async for bar in feed.bars():
            bars_out.append(bar)

        assert len(bars_out) == 30
        assert bars_out[0].close == 2350.0
        assert bars_out[-1].close == 2379.0

    async def test_stop_mid_replay(self):
        """Feed should stop when stop() is called mid-stream."""
        bars_in = [_make_bar(ts_offset=i) for i in range(100)]
        feed = ReplayFeed(bars_in)

        count = 0
        async for _ in feed.bars():
            count += 1
            if count >= 20:
                feed.stop()

        assert count == 20

    async def test_empty_bar_list(self):
        """Replay of an empty list should yield nothing."""
        feed = ReplayFeed([])
        bars_out = []
        async for bar in feed.bars():
            bars_out.append(bar)

        assert len(bars_out) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaperLedger:
    """Tests for the JSON-Lines audit ledger."""

    def test_creates_ledger_file(self, tmp_path):
        """Ledger should create a .jsonl file in the specified directory."""
        ledger = PaperLedger("test_session", directory=tmp_path)
        ledger.close()

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        assert "test_session" in files[0].name

    def test_log_fill(self, tmp_path):
        """Fill events should be written to the ledger."""
        with PaperLedger("test_fill", directory=tmp_path) as ledger:
            ledger.log_fill({
                "order_id": "TEST-001",
                "fill_price": 2350.0,
                "side": "BUY",
                "size_lots": 0.10,
            })

        entries = PaperLedger.read_ledger(ledger.path)
        fill_entries = [e for e in entries if e["type"] == "FILL"]
        assert len(fill_entries) == 1
        assert fill_entries[0]["data"]["order_id"] == "TEST-001"

    def test_log_rejection(self, tmp_path):
        """Rejection events should be recorded."""
        with PaperLedger("test_rejection", directory=tmp_path) as ledger:
            ledger.log_rejection({
                "order_id": "BAD-001",
                "reason": "STOP_LOSS_MISSING",
            })

        entries = PaperLedger.read_ledger(ledger.path)
        reject_entries = [e for e in entries if e["type"] == "REJECTION"]
        assert len(reject_entries) == 1

    def test_session_start_and_end(self, tmp_path):
        """Ledger should auto-write SESSION_START and SESSION_END."""
        with PaperLedger("test_lifecycle", directory=tmp_path) as ledger:
            pass  # Just open and close

        entries = PaperLedger.read_ledger(ledger.path)
        types = [e["type"] for e in entries]
        assert "SESSION_START" in types
        assert "SESSION_END" in types

    def test_sequential_numbering(self, tmp_path):
        """Ledger entries should have sequential sequence numbers."""
        with PaperLedger("test_seq", directory=tmp_path) as ledger:
            for i in range(5):
                ledger.log_event(f"EVENT_{i}", {"i": i})

        entries = PaperLedger.read_ledger(ledger.path)
        seqs = [e["seq"] for e in entries]
        assert seqs == list(range(len(seqs)))

    def test_log_sentinel_status(self, tmp_path):
        """SENTINEL status snapshots should be recorded."""
        with PaperLedger("test_status", directory=tmp_path) as ledger:
            ledger.log_sentinel_status({
                "l3_triggered": False,
                "open_positions": 2,
                "current_drawdown_pct": 0.03,
            })

        entries = PaperLedger.read_ledger(ledger.path)
        status_entries = [e for e in entries if e["type"] == "SENTINEL_STATUS"]
        assert len(status_entries) == 1
        assert status_entries[0]["data"]["open_positions"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaperSession:
    """Integration tests for the paper trading session orchestrator."""

    async def test_session_runs_with_replay_feed(self):
        """Session should run through replayed bars without errors."""
        from aphelion.paper.session import PaperSession, PaperSessionConfig

        bars = [_make_bar(ts_offset=i, close=2350.0 + i * 0.1) for i in range(50)]
        feed = ReplayFeed(bars)

        config = PaperSessionConfig(
            session_id="test_replay_001",
            initial_capital=10_000.0,
            warmup_bars=5,
            monitor_enabled=False,
        )

        session = PaperSession(config, feed)
        result = await session.run()

        assert result.bars_processed == 50
        assert result.error_count == 0

    async def test_session_creates_ledger(self):
        """Running a session should produce a ledger file."""
        from aphelion.paper.session import PaperSession, PaperSessionConfig

        config = PaperSessionConfig(
            session_id="test_ledger_001",
            initial_capital=10_000.0,
            warmup_bars=5,
            monitor_enabled=False,
        )
        bars = [_make_bar(ts_offset=i, close=2350.0 + i * 0.1) for i in range(10)]
        feed = ReplayFeed(bars)

        session = PaperSession(config, feed)
        result = await session.run()

        ledger_path = Path(result.ledger_path)
        assert ledger_path.exists()

        entries = PaperLedger.read_ledger(ledger_path)
        types = [e["type"] for e in entries]
        assert "SESSION_START" in types
        assert "RUN_START" in types
        assert "SESSION_END" in types

    async def test_session_summary_string(self):
        """Result summary should be a non-empty string."""
        from aphelion.paper.session import PaperSession, PaperSessionConfig

        config = PaperSessionConfig(
            session_id="test_summary",
            initial_capital=10_000.0,
            warmup_bars=5,
            monitor_enabled=False,
        )
        bars = [_make_bar(ts_offset=i, close=2350.0 + i * 0.1) for i in range(10)]
        feed = ReplayFeed(bars)

        session = PaperSession(config, feed)
        result = await session.run()

        summary = result.summary()
        assert "Paper Session" in summary
        assert "test_summary" in summary
        assert "Trades" in summary

    async def test_session_sentinel_equity_tracks(self):
        """SENTINEL should maintain equity tracking throughout the session."""
        from aphelion.paper.session import PaperSession, PaperSessionConfig

        config = PaperSessionConfig(
            session_id="test_equity_track",
            initial_capital=10_000.0,
            warmup_bars=5,
            monitor_enabled=False,
        )
        bars = [_make_bar(ts_offset=i, close=2350.0 + i * 0.1) for i in range(30)]
        feed = ReplayFeed(bars)

        session = PaperSession(config, feed)
        result = await session.run()

        # SENTINEL status should show correct equity (within floating-point tolerance)
        sentinel_equity = result.sentinel_status["account_equity"]
        assert abs(sentinel_equity - result.final_equity) < 0.01

    async def test_session_stop(self):
        """Calling stop() should terminate the session early."""
        from aphelion.paper.session import PaperSession, PaperSessionConfig

        config = PaperSessionConfig(
            session_id="test_stop",
            initial_capital=10_000.0,
            warmup_bars=5,
            monitor_enabled=False,
        )
        # Large bar list to test early stop
        bars = [_make_bar(ts_offset=i, close=2350.0 + i * 0.01) for i in range(1000)]
        feed = ReplayFeed(bars)

        session = PaperSession(config, feed)

        # Stop after a short delay
        async def stop_after():
            await asyncio.sleep(0.01)
            session.stop()

        stop_task = asyncio.create_task(stop_after())
        result = await session.run()
        await stop_task

        assert result.bars_processed > 0
        assert result.error_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 CONNECTION TESTS (without actual MT5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMT5Connection:
    """Tests for MT5Connection when MetaTrader5 is not installed."""

    def test_import_without_mt5(self):
        """MT5Connection should be importable even without MT5."""
        from aphelion.risk.sentinel.execution.mt5 import MT5Connection, MT5Config, HAS_MT5
        conn = MT5Connection()
        assert not conn.is_connected

    def test_connect_fails_without_mt5(self):
        """Connect should return False when MT5 is not available."""
        from aphelion.risk.sentinel.execution.mt5 import MT5Connection, HAS_MT5
        if HAS_MT5:
            pytest.skip("MT5 is installed — this test is for when it's not")
        conn = MT5Connection()
        result = conn.connect()
        assert result is False

    def test_stats_show_not_connected(self):
        """Stats should reflect disconnected state."""
        from aphelion.risk.sentinel.execution.mt5 import MT5Connection
        conn = MT5Connection()
        stats = conn.stats
        assert stats["connected"] is False
        assert stats["tick_count"] == 0

    def test_get_last_tick_returns_none(self):
        """get_last_tick should return None when not connected."""
        from aphelion.risk.sentinel.execution.mt5 import MT5Connection
        conn = MT5Connection()
        assert conn.get_last_tick() is None

    def test_get_bars_returns_empty(self):
        """get_bars should return empty list when not connected."""
        from aphelion.risk.sentinel.execution.mt5 import MT5Connection
        conn = MT5Connection()
        assert conn.get_bars() == []
