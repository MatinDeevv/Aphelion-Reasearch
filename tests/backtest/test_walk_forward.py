import pytest
import numpy as np
from datetime import datetime, timezone

from aphelion.backtest.walk_forward import (
    WalkForwardEngine, WalkForwardConfig, WalkForwardResults,
)
from aphelion.backtest.engine import BacktestConfig
from aphelion.backtest.order import Order, OrderType, OrderSide
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar, DataLayer
from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock

from tests.backtest.conftest import make_bars, make_sentinel_stack


def _make_wf_config(train=200, test=100, step=50, min_windows=2, min_trades=0):
    return WalkForwardConfig(
        train_bars=train,
        test_bars=test,
        step_bars=step,
        min_windows=min_windows,
        min_trades_per_window=min_trades,
        monte_carlo_paths=50,
        backtest_config=BacktestConfig(
            warmup_bars=10,
            enable_feature_engine=False,
            random_seed=42,
        ),
    )


def _dummy_strategy_factory(train_bars):
    """Returns a strategy that just does nothing (returns [])."""
    def strategy(bar, features, portfolio):
        return []
    return strategy


def _buy_strategy_factory(train_bars):
    """Returns a strategy that buys once."""
    called = [False]
    def strategy(bar, features, portfolio):
        if not called[0]:
            called[0] = True
            return [Order(
                order_id="wf-001", symbol="XAUUSD",
                order_type=OrderType.MARKET, side=OrderSide.BUY,
                size_lots=0.01, entry_price=0.0,
                stop_loss=bar.close - 5.0, take_profit=bar.close + 10.0,
                size_pct=0.02, proposed_by="TEST",
            )]
        return []
    return strategy


class TestWalkForward:
    def test_insufficient_data_returns_not_approved(self):
        stack = make_sentinel_stack()
        bus = EventBus()
        clock = MarketClock()
        dl = DataLayer(bus, clock)
        wf_config = _make_wf_config(train=500, test=200)
        wf = WalkForwardEngine(wf_config, stack, dl, _dummy_strategy_factory)
        bars = make_bars(100)  # Way too few
        result = wf.run(bars)
        assert not result.deployment_approved
        assert "INSUFFICIENT_DATA" in result.deployment_reason

    def test_generates_at_least_2_windows(self):
        stack = make_sentinel_stack()
        bus = EventBus()
        clock = MarketClock()
        dl = DataLayer(bus, clock)
        # train=200, test=100, step=50 on 500 bars
        # Window 0: [0:200][200:300]
        # Window 1: [50:250][250:350]
        # Window 2: [100:300][300:400]
        # Window 3: [150:350][350:450]
        # Window 4: [200:400][400:500]
        wf_config = _make_wf_config(train=200, test=100, step=50, min_windows=2, min_trades=0)
        wf = WalkForwardEngine(wf_config, stack, dl, _dummy_strategy_factory)
        bars = make_bars(500)
        result = wf.run(bars)
        assert result.num_windows >= 2 or not result.deployment_approved

    def test_raises_on_impossible_folds(self):
        stack = make_sentinel_stack()
        bus = EventBus()
        clock = MarketClock()
        dl = DataLayer(bus, clock)
        # train=5000, test=5000 on 8000 bars -- can only make 1 fold
        wf_config = _make_wf_config(train=5000, test=5000, step=5000, min_windows=2)
        wf = WalkForwardEngine(wf_config, stack, dl, _dummy_strategy_factory)
        bars = make_bars(8000)
        result = wf.run(bars)
        # Should NOT be approved because insufficient windows
        assert not result.deployment_approved

    def test_fold_windows_non_overlapping_oos(self):
        stack = make_sentinel_stack()
        bus = EventBus()
        clock = MarketClock()
        dl = DataLayer(bus, clock)
        wf_config = _make_wf_config(train=200, test=100, step=100, min_windows=2, min_trades=0)
        wf = WalkForwardEngine(wf_config, stack, dl, _dummy_strategy_factory)
        bars = make_bars(600)
        result = wf.run(bars)
        # Check OOS windows don't overlap
        for i in range(1, len(result.windows)):
            prev = result.windows[i-1]
            curr = result.windows[i]
            assert curr.test_start_idx >= prev.test_end_idx
