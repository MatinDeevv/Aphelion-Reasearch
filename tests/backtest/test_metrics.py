"""
Phase 3 acceptance: dedicated test suite for aphelion.backtest.metrics.
Covers all 16 standalone functions, compute_metrics integration,
_deployment_check gate, _profitability_score composite, and edge cases.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from aphelion.backtest.order import BacktestTrade, OrderSide
from aphelion.backtest.metrics import (
    sharpe_ratio,
    deflated_sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    profit_factor,
    max_drawdown,
    expectancy,
    win_rate,
    avg_risk_reward,
    avg_win_loss,
    consecutive_stats,
    monthly_pnl,
    hourly_pnl,
    exit_reason_breakdown,
    compute_metrics,
    BacktestMetrics,
    _deployment_check,
    _profitability_score,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _trade(
    net_pnl: float,
    direction: str = "LONG",
    entry_price: float = 2850.0,
    exit_price: float = 2860.0,
    stop_loss: float = 2840.0,
    commission: float = 7.0,
    exit_reason: str = "TP",
    entry_time: datetime | None = None,
    exit_time: datetime | None = None,
    bars_held: int = 10,
) -> BacktestTrade:
    entry_time = entry_time or datetime(2024, 3, 15, 10, 0, tzinfo=timezone.utc)
    exit_time = exit_time or datetime(2024, 3, 15, 10, 10, tzinfo=timezone.utc)
    return BacktestTrade(
        trade_id="BT-TEST",
        symbol="XAUUSD",
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        size_lots=0.10,
        size_pct=0.02,
        stop_loss=stop_loss,
        take_profit=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        gross_pnl=net_pnl + commission,
        commission=commission,
        net_pnl=net_pnl,
        exit_reason=exit_reason,
        bars_held=bars_held,
        proposed_by="TEST",
        entry_bar_index=0,
        exit_bar_index=bars_held,
    )


def _winning_trades(n: int = 5, pnl: float = 100.0) -> list[BacktestTrade]:
    return [_trade(pnl, exit_reason="TP") for _ in range(n)]


def _losing_trades(n: int = 5, pnl: float = -50.0) -> list[BacktestTrade]:
    return [_trade(pnl, exit_reason="SL") for _ in range(n)]


def _mixed_trades() -> list[BacktestTrade]:
    """5 wins ($100) + 3 losses ($-50) = net $350."""
    return _winning_trades(5, 100.0) + _losing_trades(3, -50.0)


# ═══════════════════════════════════════════════════════════════════════════
# sharpe_ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestSharpeRatio:
    def test_empty_returns_zero(self):
        assert sharpe_ratio([]) == 0.0

    def test_single_return_zero(self):
        assert sharpe_ratio([0.01]) == 0.0

    def test_constant_returns_zero(self):
        """Identical returns → zero std → zero Sharpe."""
        assert sharpe_ratio([0.001] * 100) == 0.0

    def test_positive_returns_positive_sharpe(self):
        rets = [0.005 + i * 0.0001 for i in range(100)]
        s = sharpe_ratio(rets)
        assert s > 0

    def test_negative_returns_negative_sharpe(self):
        rets = [-0.005 - i * 0.0001 for i in range(100)]
        s = sharpe_ratio(rets)
        assert s < 0

    def test_all_zero_returns_zero(self):
        assert sharpe_ratio([0.0] * 50) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# deflated_sharpe_ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestDeflatedSharpe:
    def test_single_trial_no_penalty(self):
        """Single trial should return a DSR close to the CDF of the Sharpe."""
        dsr = deflated_sharpe_ratio(2.0, 1, 500)
        assert 0.0 < dsr <= 1.0

    def test_many_trials_deflates(self):
        """More trials should lower DSR (multiple-testing penalty)."""
        dsr_1 = deflated_sharpe_ratio(1.5, 1, 500)
        dsr_100 = deflated_sharpe_ratio(1.5, 100, 500)
        assert dsr_100 < dsr_1

    def test_invalid_inputs_return_zero(self):
        assert deflated_sharpe_ratio(1.0, 0, 500) == 0.0
        assert deflated_sharpe_ratio(1.0, 5, 1) == 0.0

    def test_zero_sharpe_low_dsr(self):
        dsr = deflated_sharpe_ratio(0.0, 10, 500)
        assert dsr < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# sortino_ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestSortinoRatio:
    def test_empty_returns_zero(self):
        assert sortino_ratio([]) == 0.0

    def test_all_positive_returns_capped(self):
        """No downside deviation → capped finite value."""
        s = sortino_ratio([0.01, 0.02, 0.03, 0.04, 0.05])
        assert np.isfinite(s) and s >= 0

    def test_negative_returns_finite(self):
        s = sortino_ratio([-0.01, -0.02, -0.03, 0.01])
        assert np.isfinite(s)

    def test_single_return_zero(self):
        assert sortino_ratio([0.01]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# calmar_ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestCalmarRatio:
    def test_zero_drawdown_returns_zero(self):
        assert calmar_ratio(50.0, 0.0) == 0.0

    def test_zero_years_returns_zero(self):
        assert calmar_ratio(50.0, 10.0, years=0.0) == 0.0

    def test_positive_calmar(self):
        c = calmar_ratio(100.0, 10.0, years=2.0)
        assert c == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════════
# omega_ratio
# ═══════════════════════════════════════════════════════════════════════════

class TestOmegaRatio:
    def test_empty_returns_zero(self):
        assert omega_ratio([]) == 0.0

    def test_all_above_threshold_inf(self):
        o = omega_ratio([0.01, 0.02, 0.03])
        assert o == float("inf")

    def test_all_below_threshold_zero(self):
        o = omega_ratio([-0.01, -0.02, -0.03])
        assert o == 0.0

    def test_mixed_returns_finite(self):
        o = omega_ratio([0.02, -0.01, 0.03, -0.005])
        assert 0 < o < float("inf")

    def test_custom_threshold(self):
        o = omega_ratio([0.01, 0.02, 0.03], threshold=0.05)
        # All below 0.05 → zero gains → 0
        assert o == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# profit_factor
# ═══════════════════════════════════════════════════════════════════════════

class TestProfitFactor:
    def test_no_trades_zero(self):
        assert profit_factor([]) == 0.0

    def test_all_winners_inf(self):
        pf = profit_factor(_winning_trades(3))
        assert pf == float("inf")

    def test_all_losers_zero(self):
        pf = profit_factor(_losing_trades(3))
        assert pf == 0.0

    def test_mixed_correct(self):
        trades = _winning_trades(3, 100.0) + _losing_trades(2, -50.0)
        pf = profit_factor(trades)
        assert pf == pytest.approx(300.0 / 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# max_drawdown
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxDrawdown:
    def test_single_element(self):
        mdd, p, t = max_drawdown([100])
        assert mdd == 0.0

    def test_monotonic_increase_zero_dd(self):
        mdd, _, _ = max_drawdown([100, 110, 120, 130])
        assert mdd == 0.0

    def test_simple_drawdown(self):
        mdd, peak, trough = max_drawdown([100, 120, 90, 110])
        assert mdd == pytest.approx(0.25)  # 30 / 120
        assert peak == 1
        assert trough == 2

    def test_empty_equity_curve(self):
        mdd, _, _ = max_drawdown([])
        assert mdd == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# expectancy
# ═══════════════════════════════════════════════════════════════════════════

class TestExpectancy:
    def test_empty_zero(self):
        assert expectancy([]) == 0.0

    def test_all_winners_positive(self):
        e = expectancy(_winning_trades(5, 100.0))
        assert e == pytest.approx(100.0)

    def test_mixed_positive(self):
        trades = _winning_trades(3, 100.0) + _losing_trades(2, -50.0)
        e = expectancy(trades)
        assert e == pytest.approx(200.0 / 5)


# ═══════════════════════════════════════════════════════════════════════════
# win_rate
# ═══════════════════════════════════════════════════════════════════════════

class TestWinRate:
    def test_empty_zero(self):
        assert win_rate([]) == 0.0

    def test_all_winners(self):
        assert win_rate(_winning_trades(3)) == 1.0

    def test_mixed(self):
        trades = _winning_trades(3) + _losing_trades(2)
        assert win_rate(trades) == pytest.approx(0.6)


# ═══════════════════════════════════════════════════════════════════════════
# avg_win_loss
# ═══════════════════════════════════════════════════════════════════════════

class TestAvgWinLoss:
    def test_no_trades(self):
        assert avg_win_loss([]) == (0.0, 0.0)

    def test_only_winners(self):
        w, l = avg_win_loss(_winning_trades(3, 100.0))
        assert w == pytest.approx(100.0)
        assert l == 0.0

    def test_only_losers(self):
        w, l = avg_win_loss(_losing_trades(3, -50.0))
        assert w == 0.0
        assert l == pytest.approx(-50.0)


# ═══════════════════════════════════════════════════════════════════════════
# consecutive_stats
# ═══════════════════════════════════════════════════════════════════════════

class TestConsecutiveStats:
    def test_no_trades(self):
        assert consecutive_stats([]) == (0, 0)

    def test_all_wins(self):
        w, l = consecutive_stats(_winning_trades(5))
        assert w == 5
        assert l == 0

    def test_all_losses(self):
        w, l = consecutive_stats(_losing_trades(5))
        assert w == 0
        assert l == 5

    def test_alternating(self):
        trades = []
        for i in range(6):
            trades.append(_trade(100.0 if i % 2 == 0 else -50.0))
        w, l = consecutive_stats(trades)
        assert w == 1
        assert l == 1

    def test_flat_trade_resets_streak(self):
        trades = _winning_trades(2) + [_trade(0.0)] + _winning_trades(2)
        w, _ = consecutive_stats(trades)
        assert w == 2  # Flat breaks the streak


# ═══════════════════════════════════════════════════════════════════════════
# monthly_pnl / hourly_pnl / exit_reason_breakdown
# ═══════════════════════════════════════════════════════════════════════════

class TestBreakdowns:
    def test_monthly_pnl_groups_by_exit_month(self):
        t1 = _trade(100.0, exit_time=datetime(2024, 1, 15, tzinfo=timezone.utc))
        t2 = _trade(200.0, exit_time=datetime(2024, 1, 20, tzinfo=timezone.utc))
        t3 = _trade(-50.0, exit_time=datetime(2024, 2, 10, tzinfo=timezone.utc))
        result = monthly_pnl([t1, t2, t3])
        assert result["2024-01"] == pytest.approx(300.0)
        assert result["2024-02"] == pytest.approx(-50.0)

    def test_hourly_pnl_groups_by_entry_hour(self):
        t1 = _trade(100.0, entry_time=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc))
        t2 = _trade(-30.0, entry_time=datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc))
        t3 = _trade(50.0, entry_time=datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc))
        result = hourly_pnl([t1, t2, t3])
        assert result[10] == pytest.approx(70.0)
        assert result[14] == pytest.approx(50.0)

    def test_exit_reason_breakdown_counts(self):
        trades = (
            [_trade(100.0, exit_reason="TP")] * 3
            + [_trade(-50.0, exit_reason="SL")] * 2
        )
        result = exit_reason_breakdown(trades)
        assert result["TP"] == 3
        assert result["SL"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# compute_metrics (integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def _build_inputs(self):
        trades = _mixed_trades()
        equity = [10_000, 10_100, 10_200, 10_150, 10_300, 10_350,
                  10_250, 10_350, 10_300]
        daily_returns = [0.01, 0.01, -0.005, 0.015, 0.005, -0.01, 0.01, -0.005]
        return trades, equity, daily_returns

    def test_returns_backtest_metrics_dataclass(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr, initial_capital=10_000.0)
        assert isinstance(m, BacktestMetrics)

    def test_total_trades_count(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr)
        assert m.total_trades == 8

    def test_final_equity_from_curve(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr, initial_capital=10_000.0)
        assert m.final_equity == eq[-1]

    def test_total_return_pct(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr, initial_capital=10_000.0)
        expected = (eq[-1] - 10_000.0) / 10_000.0 * 100
        assert m.total_return_pct == pytest.approx(expected)

    def test_max_drawdown_populated(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr)
        assert m.max_drawdown_pct >= 0

    def test_gross_profit_and_loss(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr)
        assert m.gross_profit == pytest.approx(500.0)
        assert m.gross_loss == pytest.approx(150.0)

    def test_net_profit(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr)
        assert m.net_profit == pytest.approx(350.0)

    def test_risk_adjusted_ratios_finite(self):
        trades, eq, dr = self._build_inputs()
        m = compute_metrics(trades, eq, dr)
        assert np.isfinite(m.sharpe)
        assert np.isfinite(m.sortino)
        assert np.isfinite(m.calmar)

    def test_empty_trades_no_error(self):
        m = compute_metrics([], [10_000], [], initial_capital=10_000)
        assert m.total_trades == 0
        assert m.win_rate_pct == 0.0

    def test_broker_stats_override_commission(self):
        trades, eq, dr = self._build_inputs()
        bs = {"total_commission": 99.0, "total_slippage_cost": 10.0}
        m = compute_metrics(trades, eq, dr, broker_stats=bs)
        assert m.total_commission == pytest.approx(99.0)
        assert m.total_slippage == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════════════════
# _deployment_check
# ═══════════════════════════════════════════════════════════════════════════

class TestDeploymentCheck:
    def _passing_metrics(self) -> BacktestMetrics:
        return BacktestMetrics(
            total_trades=600,
            sharpe=1.5,
            max_drawdown_pct=10.0,
            win_rate_pct=55.0,
            profit_factor=1.8,
            net_profit=5000.0,
            expectancy_dollars=8.0,
            expectancy_r=0.25,
            payoff_ratio=1.5,
            return_over_max_drawdown=2.0,
            profitable_month_ratio=70.0,
            cost_to_gross_profit_pct=10.0,
            profitability_score=0.65,
        )

    def test_approved_when_all_pass(self):
        approved, reason = _deployment_check(self._passing_metrics())
        assert approved is True
        assert reason == "APPROVED"

    def test_rejected_too_few_trades(self):
        m = self._passing_metrics()
        m.total_trades = 100
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "TOO_FEW_TRADES" in reason

    def test_rejected_low_sharpe(self):
        m = self._passing_metrics()
        m.sharpe = 0.5
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "LOW_SHARPE" in reason

    def test_rejected_high_drawdown(self):
        m = self._passing_metrics()
        m.max_drawdown_pct = 20.0
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "HIGH_MDD" in reason

    def test_rejected_low_win_rate(self):
        m = self._passing_metrics()
        m.win_rate_pct = 40.0
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "LOW_WIN_RATE" in reason

    def test_rejected_low_profit_factor(self):
        m = self._passing_metrics()
        m.profit_factor = 1.0
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "LOW_PF" in reason

    def test_rejected_non_positive_net_profit(self):
        m = self._passing_metrics()
        m.net_profit = -100.0
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "NON_POSITIVE_NET_PROFIT" in reason

    def test_multiple_rejection_reasons(self):
        m = self._passing_metrics()
        m.sharpe = 0.1
        m.total_trades = 10
        approved, reason = _deployment_check(m)
        assert approved is False
        assert "LOW_SHARPE" in reason
        assert "TOO_FEW_TRADES" in reason


# ═══════════════════════════════════════════════════════════════════════════
# _profitability_score
# ═══════════════════════════════════════════════════════════════════════════

class TestProfitabilityScore:
    def test_zero_when_too_few_trades(self):
        m = BacktestMetrics(total_trades=10, net_profit=100.0, expectancy_dollars=5.0)
        assert _profitability_score(m) == 0.0

    def test_zero_when_negative_net_profit(self):
        m = BacktestMetrics(total_trades=100, net_profit=-50.0, expectancy_dollars=-1.0)
        assert _profitability_score(m) == 0.0

    def test_zero_when_negative_expectancy(self):
        m = BacktestMetrics(total_trades=100, net_profit=100.0, expectancy_dollars=-1.0)
        assert _profitability_score(m) == 0.0

    def test_bounded_zero_to_one(self):
        m = BacktestMetrics(
            total_trades=100,
            net_profit=5000.0,
            expectancy_dollars=50.0,
            profit_factor=2.5,
            expectancy_r=1.0,
            return_over_max_drawdown=2.0,
            win_rate_pct=60.0,
            profitable_month_ratio=75.0,
            cost_to_gross_profit_pct=5.0,
        )
        score = _profitability_score(m)
        assert 0 < score <= 1.0

    def test_higher_metrics_higher_score(self):
        m_good = BacktestMetrics(
            total_trades=100, net_profit=5000.0, expectancy_dollars=50.0,
            profit_factor=2.5, expectancy_r=1.0, return_over_max_drawdown=2.0,
            win_rate_pct=60.0, profitable_month_ratio=75.0, cost_to_gross_profit_pct=5.0,
        )
        m_bad = BacktestMetrics(
            total_trades=100, net_profit=100.0, expectancy_dollars=1.0,
            profit_factor=1.1, expectancy_r=0.1, return_over_max_drawdown=0.5,
            win_rate_pct=42.0, profitable_month_ratio=30.0, cost_to_gross_profit_pct=50.0,
        )
        assert _profitability_score(m_good) > _profitability_score(m_bad)
