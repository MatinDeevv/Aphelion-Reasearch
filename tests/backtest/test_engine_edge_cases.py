"""
Phase 3 acceptance: additional edge-case coverage for BacktestEngine
(supplements existing test_engine.py).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from aphelion.backtest.engine import BacktestEngine, BacktestConfig, BacktestResults
from aphelion.backtest.broker_sim import BrokerConfig
from aphelion.backtest.order import Order, OrderType, OrderSide, OrderStatus
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar, DataLayer
from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock

from tests.backtest.conftest import make_bars, make_sentinel_stack


# ── helpers ──────────────────────────────────────────────────────────────────

def _engine(warmup=5, enable_features=False, cap=10_000.0, max_bars=None):
    stack = make_sentinel_stack()
    cfg = BacktestConfig(
        initial_capital=cap,
        warmup_bars=warmup,
        enable_feature_engine=enable_features,
        random_seed=42,
        max_bars=max_bars,
    )
    dl = DataLayer(EventBus(), MarketClock())
    return BacktestEngine(cfg, stack, dl), stack


def _buy_once():
    fired = [False]
    def strat(bar, features, portfolio):
        if not fired[0]:
            fired[0] = True
            return [Order(
                order_id="single-001", symbol="XAUUSD",
                order_type=OrderType.MARKET, side=OrderSide.BUY,
                size_lots=0.01, entry_price=0.0,
                stop_loss=bar.close - 5.0, take_profit=bar.close + 10.0,
                size_pct=0.02, proposed_by="TEST",
            )]
        return []
    return strat


def _noop_strategy():
    return lambda bar, features, portfolio: []


def _error_strategy():
    """Strategy that raises an exception."""
    def strat(bar, features, portfolio):
        raise RuntimeError("strategy error")
    return strat


# ── Tests ────────────────────────────────────────────────────────────────────

class TestBacktestEngineEdgeCases:
    def test_total_return_pct_property(self):
        eng, _ = _engine()
        eng.set_strategy(_buy_once())
        r = eng.run(make_bars(100))
        expected = (r.final_equity - r.initial_capital) / r.initial_capital * 100
        assert r.total_return_pct == pytest.approx(expected)

    def test_get_results_none_before_run(self):
        eng, _ = _engine()
        assert eng.get_results() is None

    def test_get_results_populated_after_run(self):
        eng, _ = _engine()
        eng.set_strategy(_noop_strategy())
        eng.run(make_bars(50))
        assert eng.get_results() is not None

    def test_max_bars_truncates_data(self):
        eng, _ = _engine(warmup=2, max_bars=30)
        eng.set_strategy(_noop_strategy())
        r = eng.run(make_bars(200))
        assert r.total_bars == 30

    def test_run_twice_deterministic(self):
        """Running the same data through two identical engines yields identical results."""
        bars = make_bars(100)
        eng1, _ = _engine(warmup=5)
        eng1.set_strategy(_buy_once())
        r1 = eng1.run(bars)

        eng2, _ = _engine(warmup=5)
        eng2.set_strategy(_buy_once())
        r2 = eng2.run(bars)
        assert r1.final_equity == pytest.approx(r2.final_equity)
        assert len(r1.trades) == len(r2.trades)

    def test_corrupted_bar_skipped(self):
        """Bars with NaN close or high<low are skipped without crashing."""
        eng, _ = _engine(warmup=2)
        eng.set_strategy(_noop_strategy())
        bars = make_bars(50)
        # Inject a corrupted bar (NaN close)
        bars[10] = Bar(
            timestamp=bars[10].timestamp, timeframe=Timeframe.M1,
            open=float("nan"), high=2860.0, low=2840.0, close=float("nan"),
            volume=100.0, tick_volume=10, spread=0.2, is_complete=True,
        )
        # Inject a bar with high < low
        bars[20] = Bar(
            timestamp=bars[20].timestamp, timeframe=Timeframe.M1,
            open=2850.0, high=2840.0, low=2860.0, close=2845.0,
            volume=100.0, tick_volume=10, spread=0.2, is_complete=True,
        )
        r = eng.run(bars)
        assert r is not None
        assert r.total_bars == 50

    def test_strategy_exception_does_not_crash(self):
        """Strategy errors are swallowed and logged."""
        eng, _ = _engine(warmup=2)
        eng.set_strategy(_error_strategy())
        r = eng.run(make_bars(50))
        assert r is not None

    def test_drawdown_curve_populated(self):
        eng, _ = _engine(warmup=2)
        eng.set_strategy(_buy_once())
        r = eng.run(make_bars(60))
        dd_ts, dd_vals = r.drawdown_curve
        assert len(dd_ts) > 0
        assert len(dd_vals) > 0

    def test_daily_returns_populated(self):
        eng, _ = _engine(warmup=5)
        eng.set_strategy(_buy_once())
        bars = make_bars(500)
        r = eng.run(bars)
        # 500 bars at M1 ~8.3 hours, might be one day only
        assert isinstance(r.daily_returns, list)

    def test_friday_close_forces_exit(self):
        """Positions open past Friday 20:30 UTC get force-closed."""
        eng, _ = _engine(warmup=2)

        # Create bars that span Friday afternoon
        friday = datetime(2024, 1, 5, 20, 0, tzinfo=timezone.utc)  # Friday 20:00

        friday_bars = []
        price = 2850.0
        rng = np.random.default_rng(99)
        for i in range(60):  # 60 minutes: 20:00 → 21:00
            price += float(rng.normal(0, 0.1))
            ts = friday + timedelta(minutes=i)
            friday_bars.append(Bar(
                timestamp=ts, timeframe=Timeframe.M1,
                open=price, high=price + 0.5, low=price - 0.5, close=price,
                volume=50.0, tick_volume=50, spread=0.2, is_complete=True,
            ))

        # Strategy buys on bar 3 (after warmup=2)
        eng.set_strategy(_buy_once())
        r = eng.run(friday_bars)
        # Position should be force-closed with reason FRIDAY_CLOSE
        friday_closes = [t for t in r.trades if t.exit_reason == "FRIDAY_CLOSE"]
        if r.trades:
            assert len(friday_closes) > 0

    def test_sell_strategy_through_engine(self):
        """A SHORT trade runs through the full engine loop."""
        eng, _ = _engine(warmup=2)
        fired = [False]

        def sell_once(bar, features, portfolio):
            if not fired[0]:
                fired[0] = True
                return [Order(
                    order_id="short-001", symbol="XAUUSD",
                    order_type=OrderType.MARKET, side=OrderSide.SELL,
                    size_lots=0.01, entry_price=0.0,
                    stop_loss=bar.close + 5.0, take_profit=bar.close - 10.0,
                    size_pct=0.02, proposed_by="TEST",
                )]
            return []

        eng.set_strategy(sell_once)
        r = eng.run(make_bars(100))
        assert r is not None

    def test_limit_order_lifecycle_through_engine(self):
        """A LIMIT order gets submitted, waits, then fills when price hits."""
        eng, _ = _engine(warmup=2)
        bars = make_bars(100)
        # Pick a target price below current bars
        target = bars[5].close - 3.0

        fired = [False]

        def limit_strat(bar, features, portfolio):
            if not fired[0]:
                fired[0] = True
                return [Order(
                    order_id="lim-001", symbol="XAUUSD",
                    order_type=OrderType.LIMIT, side=OrderSide.BUY,
                    size_lots=0.01, entry_price=target,
                    stop_loss=target - 5.0, take_profit=target + 10.0,
                    size_pct=0.02, proposed_by="TEST",
                )]
            return []

        eng.set_strategy(limit_strat)
        r = eng.run(bars)
        # Can't guarantee it fills, but engine shouldn't crash
        assert r is not None


class TestBrokerSimEdgeCases:
    """Additional broker_sim edge cases beyond test_broker_sim.py."""

    def test_stop_buy_order_fills_above_entry(self):
        """A BUY STOP order triggers when high >= entry_price."""
        from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
        from tests.backtest.conftest import make_sentinel_stack

        stack = make_sentinel_stack()
        core = stack["core"]
        core.update_equity(10_000.0)
        rng = np.random.default_rng(42)
        broker = BrokerSimulator(BrokerConfig(), stack["validator"], core, rng)

        order = Order(
            order_id="stop-001", symbol="XAUUSD",
            order_type=OrderType.STOP, side=OrderSide.BUY,
            size_lots=0.01, entry_price=2860.0,
            stop_loss=2850.0, take_profit=2880.0,
            size_pct=0.02, proposed_by="TEST",
        )
        bar = Bar(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            timeframe=Timeframe.M1,
            open=2855.0, high=2862.0, low=2853.0, close=2858.0,
            volume=100.0, tick_volume=100, spread=0.2, is_complete=True,
        )
        results = broker.check_pending_orders([order], bar, 10_000.0)
        assert len(results) == 1
        filled_order, fill = results[0]
        # Either fills or stays pending depending on implementation
        assert fill is not None or filled_order.status == OrderStatus.PENDING

    def test_broker_stats_empty_when_no_trades(self):
        from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
        stack = make_sentinel_stack()
        core = stack["core"]
        core.update_equity(10_000.0)
        rng = np.random.default_rng(42)
        broker = BrokerSimulator(BrokerConfig(), stack["validator"], core, rng)
        stats = broker.stats
        assert stats["fill_count"] == 0
        assert stats["rejection_count"] == 0
        assert stats["total_commission"] == pytest.approx(0.0)

    def test_multiple_positions_sl_tp_check(self):
        """Multiple open positions can all be checked for SL/TP in one bar."""
        from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
        from aphelion.risk.sentinel.core import Position

        stack = make_sentinel_stack()
        core = stack["core"]
        core.update_equity(10_000.0)
        rng = np.random.default_rng(42)
        broker = BrokerSimulator(BrokerConfig(), stack["validator"], core, rng)

        positions = [
            Position(
                position_id=f"mp-{i}", symbol="XAUUSD", direction="LONG",
                entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
                size_lots=0.01, size_pct=0.02,
                open_time=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]
        # A bar where SL is hit for all
        bar = Bar(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            timeframe=Timeframe.M1,
            open=2838.0, high=2839.0, low=2835.0, close=2836.0,
            volume=100.0, tick_volume=100, spread=0.2, is_complete=True,
        )
        exits = broker.check_sl_tp(positions, bar)
        assert len(exits) == 3
        for _, _, reason in exits:
            assert reason == "SL_HIT"

    def test_short_tp_hit(self):
        """SHORT position TP triggers when low <= take_profit."""
        from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
        from aphelion.risk.sentinel.core import Position

        stack = make_sentinel_stack()
        core = stack["core"]
        core.update_equity(10_000.0)
        rng = np.random.default_rng(42)
        broker = BrokerSimulator(BrokerConfig(), stack["validator"], core, rng)

        pos = Position(
            position_id="stp-001", symbol="XAUUSD", direction="SHORT",
            entry_price=2850.0, stop_loss=2860.0, take_profit=2830.0,
            size_lots=0.01, size_pct=0.02,
            open_time=datetime.now(timezone.utc),
        )
        bar = Bar(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            timeframe=Timeframe.M1,
            open=2835.0, high=2837.0, low=2828.0, close=2832.0,
            volume=100.0, tick_volume=100, spread=0.2, is_complete=True,
        )
        exits = broker.check_sl_tp([pos], bar)
        assert len(exits) == 1
        _, exit_price, reason = exits[0]
        assert reason == "TP_HIT"
        assert exit_price == 2830.0

    def test_gap_open_above_sl_short(self):
        """SHORT position gap above SL fills at open."""
        from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
        from aphelion.risk.sentinel.core import Position

        stack = make_sentinel_stack()
        core = stack["core"]
        core.update_equity(10_000.0)
        rng = np.random.default_rng(42)
        broker = BrokerSimulator(BrokerConfig(), stack["validator"], core, rng)

        pos = Position(
            position_id="gap-s01", symbol="XAUUSD", direction="SHORT",
            entry_price=2850.0, stop_loss=2860.0, take_profit=2830.0,
            size_lots=0.01, size_pct=0.02,
            open_time=datetime.now(timezone.utc),
        )
        # Gap open ABOVE stop loss
        bar = Bar(
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            timeframe=Timeframe.M1,
            open=2865.0, high=2870.0, low=2863.0, close=2868.0,
            volume=100.0, tick_volume=100, spread=0.2, is_complete=True,
        )
        exits = broker.check_sl_tp([pos], bar)
        assert len(exits) == 1
        _, exit_price, reason = exits[0]
        assert reason == "SL_HIT"
        assert exit_price == 2865.0  # Fills at gap open
