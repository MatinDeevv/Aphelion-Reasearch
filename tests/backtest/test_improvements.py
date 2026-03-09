"""Tests for backtest Phase 4-7 improvements: R-multiple fix, broker slippage,
Sortino, DSR, Monte Carlo ruin, walk-forward embargo."""

import math
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from aphelion.backtest.order import BacktestTrade, OrderType, OrderSide, OrderStatus
from aphelion.backtest.analytics import PerformanceAnalyzer
from aphelion.backtest.metrics import (
    deflated_sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
)
from aphelion.backtest.monte_carlo import MonteCarloEngine, MonteCarloConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_trade(
    direction="LONG", entry=2850.0, exit_=2860.0,
    sl=2840.0, tp=2870.0, lots=0.01, comm=0.14,
    reason="TP_HIT", bars=10,
) -> BacktestTrade:
    now = datetime.now(timezone.utc)
    gross = (exit_ - entry) * lots * 100 if direction == "LONG" else (entry - exit_) * lots * 100
    return BacktestTrade(
        trade_id="T001", symbol="XAUUSD", direction=direction,
        entry_price=entry, exit_price=exit_, size_lots=lots, size_pct=0.02,
        stop_loss=sl, take_profit=tp, entry_time=now, exit_time=now,
        gross_pnl=gross, commission=comm, net_pnl=gross - comm,
        exit_reason=reason, bars_held=bars, proposed_by="TEST",
        entry_bar_index=0, exit_bar_index=bars,
    )


# ── R-Multiple Tests ────────────────────────────────────────────────────────

class TestRMultipleFix:
    def test_r_multiple_includes_lot_size(self):
        """R-multiple must account for the 100oz/lot multiplier."""
        t = _make_trade(entry=2850.0, exit_=2870.0, sl=2840.0, lots=0.01, comm=0.0)
        # gross = 20 * 0.01 * 100 = $20
        # risk = 10 * 0.01 * 100 = $10
        # R = 20/10 = 2.0
        assert t.r_multiple == pytest.approx(2.0, rel=0.01)

    def test_r_multiple_scale_invariant(self):
        """Same price move at different lot sizes should yield same R."""
        t1 = _make_trade(entry=2850.0, exit_=2870.0, sl=2840.0, lots=0.01, comm=0.0)
        t2 = _make_trade(entry=2850.0, exit_=2870.0, sl=2840.0, lots=1.0, comm=0.0)
        assert t1.r_multiple == pytest.approx(t2.r_multiple, rel=0.01)

    def test_r_multiple_negative_for_loss(self):
        t = _make_trade(entry=2850.0, exit_=2845.0, sl=2840.0, lots=0.01, comm=0.0)
        assert t.r_multiple < 0.0

    def test_pnl_pct_based_on_dollar_risk(self):
        """pnl_pct should be based on initial dollar risk, not magic constant."""
        t = _make_trade(entry=2850.0, exit_=2870.0, sl=2840.0, lots=0.01, comm=0.0)
        # risk_dollars = 10 * 0.01 * 100 = $10
        # pnl_pct = (20/10) * 100 = 200%
        assert t.pnl_pct == pytest.approx(200.0, rel=0.01)


# ── Broker Slippage Tests ───────────────────────────────────────────────────

class TestBrokerSlippageFix:
    """Verify gap_slippage_multiplier is NOT applied to normal SL hits."""

    def test_normal_sl_no_gap_multiplier(self):
        from aphelion.backtest.broker_sim import BrokerSimulator
        import inspect
        source = inspect.getsource(BrokerSimulator.check_sl_tp)
        # Normal SL path should NOT use gap_slippage_multiplier
        # The code raw_slip line should use exponential(cfg.slippage_pips) only
        # Count occurrences — gap_slippage_multiplier should only appear in gap-through, not normal
        lines = source.split('\n')
        normal_sl_uses_gap = False
        for line in lines:
            if 'exponential' in line and 'gap_slippage_multiplier' in line:
                normal_sl_uses_gap = True
        assert not normal_sl_uses_gap, "Normal SL still uses gap_slippage_multiplier"


# ── Sortino Fix Tests ───────────────────────────────────────────────────────

class TestSortinoFix:
    def test_sortino_uses_excess_returns_for_downside(self):
        """Verify analytics Sortino uses excess returns (not raw) for downside."""
        import inspect
        source = inspect.getsource(PerformanceAnalyzer.sortino_ratio.fget)
        # The downside array should filter from `excess`, not raw `returns`
        assert "excess[excess < 0]" in source or "downside = excess" in source

    def test_metrics_sortino_capped_not_inf(self):
        """All-positive returns should not produce inf."""
        returns = [0.005, 0.003, 0.002, 0.004, 0.001] * 20
        result = sortino_ratio(returns)
        assert result != float("inf")
        assert result <= 10.0


# ── DSR Guard Tests ─────────────────────────────────────────────────────────

class TestDSRGuard:
    def test_dsr_negative_skew_high_sharpe_no_crash(self):
        """DSR should not crash when se formula would produce negative sqrt."""
        result = deflated_sharpe_ratio(
            observed_sharpe=3.0,
            num_trials=100,
            backtest_length_days=500,
            skewness=-5.0,  # Extreme negative skew
            kurtosis=20.0,  # High kurtosis
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_dsr_normal_case(self):
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            num_trials=10,
            backtest_length_days=252,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ── Monte Carlo Ruin Tests ──────────────────────────────────────────────────

class TestMonteCarloRuin:
    def test_equity_freezes_at_zero(self):
        """After ruin (equity <= 0), path should stay at 0, not recover."""
        config = MonteCarloConfig(num_paths=50, random_seed=42)
        engine = MonteCarloEngine(config)

        # Create trades that will certainly cause ruin: all massive losses
        trades = []
        now = datetime.now(timezone.utc)
        for i in range(20):
            t = BacktestTrade(
                trade_id=f"T{i}", symbol="XAUUSD", direction="LONG",
                entry_price=2850.0, exit_price=2840.0,
                size_lots=1.0, size_pct=0.10,
                stop_loss=2840.0, take_profit=2870.0,
                entry_time=now, exit_time=now,
                gross_pnl=-1000.0, commission=7.0, net_pnl=-1007.0,
                exit_reason="SL_HIT", bars_held=5, proposed_by="TEST",
                entry_bar_index=i * 5, exit_bar_index=i * 5 + 5,
            )
            trades.append(t)

        result = engine.run(trades, initial_capital=5000.0)

        # Check that paths that hit 0 don't recover
        for path_equity in [result.p5_equity]:
            for j in range(1, len(path_equity)):
                if path_equity[j - 1] == 0.0:
                    assert path_equity[j] == 0.0, \
                        f"Equity recovered from 0 at index {j}"


# ── Analytics Score Normalization ────────────────────────────────────────────

class TestAnalyticsScoreNormalized:
    def test_score_bounded_0_to_1(self):
        """Score should always be in [0, 1] range."""
        now = datetime.now(timezone.utc)
        trades = []
        for i in range(50):
            win = i % 3 != 0  # 66% win rate
            pnl = 30.0 if win else -15.0
            trades.append(BacktestTrade(
                trade_id=f"T{i}", symbol="XAUUSD", direction="LONG",
                entry_price=2850.0,
                exit_price=2860.0 if win else 2845.0,
                size_lots=0.01, size_pct=0.02,
                stop_loss=2840.0, take_profit=2870.0,
                entry_time=now + timedelta(hours=i),
                exit_time=now + timedelta(hours=i, minutes=30),
                gross_pnl=pnl, commission=0.14, net_pnl=pnl - 0.14,
                exit_reason="TP_HIT" if win else "SL_HIT",
                bars_held=10, proposed_by="TEST",
                entry_bar_index=i * 10, exit_bar_index=i * 10 + 10,
            ))

        equity = [10000.0]
        for t in trades:
            equity.append(equity[-1] + t.net_pnl)
        timestamps = [now + timedelta(hours=i) for i in range(len(equity))]

        analyzer = PerformanceAnalyzer(trades, equity, timestamps, 10000.0)
        s = analyzer.score()
        assert 0.0 <= s <= 1.0
