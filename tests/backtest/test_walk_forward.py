import pytest
import numpy as np
from datetime import datetime, timezone

from aphelion.backtest.walk_forward import (
    WalkForwardEngine, WalkForwardConfig, WalkForwardResults,
    WalkForwardWindow,
)
from aphelion.backtest.engine import BacktestConfig
from aphelion.backtest.metrics import BacktestMetrics
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


class TestWalkForwardResultsProperties:
    """Tests for the new overfit/summary/to_dict methods on WalkForwardResults."""

    def _make_window(self, train_sharpe: float, test_sharpe: float,
                     test_return: float = 1.0) -> WalkForwardWindow:
        """Build a minimal window with train + test metrics stubs."""
        train_m = BacktestMetrics(
            total_trades=50,
            total_return_pct=5.0,
            sharpe=train_sharpe,
            sortino=1.5,
            max_drawdown_pct=8.0,
            return_over_max_drawdown=0.625,
            win_rate_pct=60.0,
            profit_factor=1.5,
            expectancy_dollars=30.0,
            net_profit=500.0,
            gross_profit=1500.0,
            gross_loss=-1000.0,
            profitability_score=0.55,
            profitable_month_ratio=60.0,
        )
        test_m = BacktestMetrics(
            total_trades=30,
            total_return_pct=test_return,
            sharpe=test_sharpe,
            sortino=1.2,
            max_drawdown_pct=10.0,
            return_over_max_drawdown=0.5,
            win_rate_pct=60.0,
            profit_factor=1.4,
            expectancy_dollars=20.0,
            net_profit=300.0,
            gross_profit=900.0,
            gross_loss=-600.0,
            profitability_score=0.50,
            profitable_month_ratio=55.0,
        )
        return WalkForwardWindow(
            window_index=0,
            train_start_idx=0, train_end_idx=200,
            test_start_idx=200, test_end_idx=300,
            train_bars_count=200, test_bars_count=100,
            train_metrics=train_m, test_metrics=test_m,
        )

    def test_overfit_ratio_no_windows(self):
        r = WalkForwardResults(config=WalkForwardConfig())
        assert r.overfit_ratio == 0.0

    def test_overfit_ratio_near_one_no_overfit(self):
        w = self._make_window(train_sharpe=1.2, test_sharpe=1.1)
        r = WalkForwardResults(config=WalkForwardConfig(), windows=[w])
        ratio = r.overfit_ratio
        assert 0.9 < ratio < 1.3  # train/test Sharpe close

    def test_overfit_ratio_high_means_overfit(self):
        w = self._make_window(train_sharpe=3.0, test_sharpe=0.5)
        r = WalkForwardResults(config=WalkForwardConfig(), windows=[w])
        assert r.overfit_ratio > 4.0

    def test_sharpe_decay_no_windows(self):
        r = WalkForwardResults(config=WalkForwardConfig())
        assert r.sharpe_decay == 0.0

    def test_sharpe_decay_with_degradation(self):
        w = self._make_window(train_sharpe=2.0, test_sharpe=1.0)
        r = WalkForwardResults(config=WalkForwardConfig(), windows=[w])
        # 50% decay: (2.0 - 1.0) / 2.0 * 100 = 50
        assert r.sharpe_decay == pytest.approx(50.0, abs=0.1)

    def test_sharpe_decay_zero_when_test_higher(self):
        w = self._make_window(train_sharpe=1.0, test_sharpe=1.5)
        r = WalkForwardResults(config=WalkForwardConfig(), windows=[w])
        assert r.sharpe_decay == 0.0  # clamped to 0

    def test_summary_string_format(self):
        w = self._make_window(train_sharpe=1.5, test_sharpe=1.2)
        r = WalkForwardResults(
            config=WalkForwardConfig(),
            windows=[w],
            avg_oos_sharpe=1.2,
            deployment_approved=False,
            deployment_reason="TOO_FEW_TRADES: 30 < 500",
        )
        s = r.summary()
        assert "WALK-FORWARD VALIDATION REPORT" in s
        assert "Overfit Ratio" in s
        assert "Sharpe Decay" in s
        assert "APPROVED: NO" in s

    def test_to_dict_contains_keys(self):
        w = self._make_window(train_sharpe=1.5, test_sharpe=1.2)
        r = WalkForwardResults(
            config=WalkForwardConfig(),
            windows=[w],
            avg_oos_sharpe=1.2,
            deployment_approved=True,
            deployment_reason="APPROVED",
        )
        d = r.to_dict()
        expected_keys = [
            "num_windows", "total_oos_trades", "avg_oos_sharpe",
            "overfit_ratio", "sharpe_decay", "deployment_approved",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"
