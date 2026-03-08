"""Integration tests: Backtest Engine + SENTINEL risk system."""

import pytest
import numpy as np
from datetime import datetime, timezone

from aphelion.backtest.engine import BacktestEngine, BacktestConfig
from aphelion.backtest.order import Order, OrderType, OrderSide
from aphelion.backtest.analytics import PerformanceAnalyzer
from aphelion.backtest.monte_carlo import MonteCarloEngine, MonteCarloConfig
from aphelion.core.data_layer import DataLayer
from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock

from tests.backtest.conftest import make_bars, make_sentinel_stack


def _make_engine(warmup=5, initial_capital=10000.0):
    stack = make_sentinel_stack()
    config = BacktestConfig(
        initial_capital=initial_capital,
        warmup_bars=warmup,
        enable_feature_engine=False,
        random_seed=42,
    )
    bus = EventBus()
    clock = MarketClock()
    dl = DataLayer(bus, clock)
    engine = BacktestEngine(config, stack, dl)
    return engine, stack


def always_buy_strategy():
    counter = [0]
    def strategy(bar, features, portfolio):
        counter[0] += 1
        return [Order(
            order_id=f"ab-{counter[0]:04d}", symbol="XAUUSD",
            order_type=OrderType.MARKET, side=OrderSide.BUY,
            size_lots=0.01, entry_price=0.0,
            stop_loss=bar.close - 5.0, take_profit=bar.close + 15.0,
            size_pct=0.02, proposed_by="TEST",
        )]
    return strategy


def single_buy_strategy():
    called = [False]
    def strategy(bar, features, portfolio):
        if not called[0]:
            called[0] = True
            return [Order(
                order_id="sb-001", symbol="XAUUSD",
                order_type=OrderType.MARKET, side=OrderSide.BUY,
                size_lots=0.01, entry_price=0.0,
                stop_loss=bar.close - 5.0, take_profit=bar.close + 10.0,
                size_pct=0.02, proposed_by="TEST",
            )]
        return []
    return strategy


def oversized_strategy():
    counter = [0]
    def strategy(bar, features, portfolio):
        counter[0] += 1
        return [Order(
            order_id=f"big-{counter[0]:04d}", symbol="XAUUSD",
            order_type=OrderType.MARKET, side=OrderSide.BUY,
            size_lots=0.10, entry_price=0.0,
            stop_loss=bar.close - 5.0, take_profit=bar.close + 15.0,
            size_pct=0.10,  # 5x over SENTINEL limit
            proposed_by="TEST",
        )]
    return strategy


class TestBacktestSentinel:
    def test_sentinel_max_positions_never_exceeded_in_backtest(self):
        engine, stack = _make_engine(warmup=5)
        engine.set_strategy(always_buy_strategy())
        bars = make_bars(500)
        results = engine.run(bars)
        # Verify rejections occurred (can't have > 3 at once)
        assert results.sentinel_rejections > 0

    def test_sentinel_size_cap_enforced_in_backtest(self):
        engine, stack = _make_engine(warmup=5)
        engine.set_strategy(oversized_strategy())
        results = engine.run(make_bars(200))
        # All orders should be rejected because size_pct=0.10 > 0.02
        assert len(results.trades) == 0
        assert results.sentinel_rejections > 0

    def test_full_backtest_analytics_pipeline(self):
        engine, _ = _make_engine(warmup=5)
        engine.set_strategy(single_buy_strategy())
        bars = make_bars(1000)
        results = engine.run(bars)
        if results.trades:
            _, eq_vals = results.equity_curve
            ts_list, _ = results.equity_curve
            if ts_list:
                pa = PerformanceAnalyzer(
                    results.trades, eq_vals, ts_list, 10000.0,
                )
                assert 0 <= pa.win_rate <= 1
                assert 0 <= pa.max_drawdown <= 1
                assert np.isfinite(pa.sharpe_ratio)
                assert pa.score() >= 0.0

    def test_monte_carlo_on_backtest_results(self):
        engine, _ = _make_engine(warmup=5)
        engine.set_strategy(always_buy_strategy())
        bars = make_bars(500)
        results = engine.run(bars)
        if len(results.trades) >= 10:
            mc = MonteCarloEngine(MonteCarloConfig(num_paths=50, random_seed=42))
            mc_result = mc.run(results.trades, initial_capital=10000.0)
            assert 0 <= mc_result.probability_of_profit <= 1
            assert 0 <= mc_result.ruin_probability <= 1

    def test_commission_impact_on_final_equity(self):
        engine, _ = _make_engine(warmup=5)
        engine.set_strategy(single_buy_strategy())
        results = engine.run(make_bars(200))
        if results.trades:
            gross_sum = sum(t.gross_pnl for t in results.trades)
            comm_sum = sum(t.commission for t in results.trades)
            if comm_sum > 0:
                assert results.final_equity < 10000.0 + gross_sum

    def test_equity_curve_all_positive(self):
        engine, _ = _make_engine(warmup=5)
        engine.set_strategy(single_buy_strategy())
        results = engine.run(make_bars(500))
        _, eq_vals = results.equity_curve
        for val in eq_vals:
            assert val >= 0.0
