"""Tests for APHELION Almgren-Chriss Optimal Execution."""

import numpy as np
import pytest

from aphelion.risk.execution.almgren_chriss import (
    AlmgrenChrissSolver,
    ExecutionConfig,
    ExecutionPlan,
    ExecutionMonitor,
    MarketImpactEstimator,
    ImpactEstimate,
)


# ─── ExecutionConfig tests ───────────────────────────────────────────────────


class TestExecutionConfig:

    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.n_intervals > 0
        assert cfg.risk_aversion > 0
        assert cfg.interval_seconds > 0

    def test_custom(self):
        cfg = ExecutionConfig(volatility=0.5, risk_aversion=1e-5, n_intervals=10)
        assert cfg.volatility == 0.5
        assert cfg.n_intervals == 10


# ─── MarketImpactEstimator tests ─────────────────────────────────────────────


class TestMarketImpactEstimator:

    def test_estimate_returns_impact(self):
        impact = MarketImpactEstimator.estimate(
            bid_ask_spread=0.30,
            avg_volume_per_interval=500_000.0,
            volatility_per_interval=0.02,
            order_size=100.0,
        )
        assert isinstance(impact, ImpactEstimate)
        assert impact.temporary_impact > 0
        assert impact.permanent_impact > 0
        assert impact.permanent_impact < impact.temporary_impact

    def test_wider_spread_increases_temporary_impact(self):
        narrow = MarketImpactEstimator.estimate(
            bid_ask_spread=0.10,
            avg_volume_per_interval=500_000.0,
            volatility_per_interval=0.02,
            order_size=100.0,
        )
        wide = MarketImpactEstimator.estimate(
            bid_ask_spread=2.00,
            avg_volume_per_interval=500_000.0,
            volatility_per_interval=0.02,
            order_size=100.0,
        )
        assert wide.temporary_impact > narrow.temporary_impact

    def test_zero_volume_handled(self):
        impact = MarketImpactEstimator.estimate(
            bid_ask_spread=0.30,
            avg_volume_per_interval=0.0,
            volatility_per_interval=0.02,
            order_size=100.0,
        )
        assert impact.temporary_impact >= 0


# ─── AlmgrenChrissSolver tests ───────────────────────────────────────────────


class TestAlmgrenChrissSolver:

    def test_solve_returns_plan(self):
        solver = AlmgrenChrissSolver()
        plan = solver.solve(
            total_lots=1000.0,
            volatility=0.02,
            temporary_impact=1e-4,
            permanent_impact=5e-5,
            risk_aversion=1e-6,
            n_intervals=20,
        )
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.trade_schedule) == 20
        assert plan.expected_cost > 0

    def test_schedule_sums_to_total(self):
        solver = AlmgrenChrissSolver()
        total = 5000.0
        plan = solver.solve(
            total_lots=total,
            volatility=0.02,
            temporary_impact=1e-4,
            permanent_impact=5e-5,
            risk_aversion=1e-6,
            n_intervals=10,
        )
        schedule_total = sum(plan.trade_schedule)
        assert abs(schedule_total - total) < 1e-4

    def test_trajectory_starts_at_total_ends_at_zero(self):
        solver = AlmgrenChrissSolver()
        plan = solver.solve(
            total_lots=1000.0,
            volatility=0.02,
            temporary_impact=1e-4,
            permanent_impact=5e-5,
            risk_aversion=1e-6,
            n_intervals=10,
        )
        assert abs(plan.position_trajectory[0] - 1000.0) < 1e-6
        assert abs(plan.position_trajectory[-1]) < 1e-6

    def test_high_urgency_front_loads(self):
        solver = AlmgrenChrissSolver()
        plan = solver.solve(
            total_lots=1000.0,
            volatility=0.02,
            temporary_impact=1e-6,
            permanent_impact=5e-5,
            risk_aversion=1e-3,
            n_intervals=20,
        )
        assert plan.is_front_loaded

    def test_solve_from_market(self):
        solver = AlmgrenChrissSolver()
        plan = solver.solve_from_market(
            total_lots=1000.0,
            bid_ask_spread=0.30,
            avg_volume=500_000.0,
            volatility=0.02,
            risk_aversion=1e-6,
        )
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.trade_schedule) > 0

    def test_adaptive_urgency(self):
        solver = AlmgrenChrissSolver()
        x_next = solver.adaptive_urgency(
            remaining_lots=500.0,
            remaining_intervals=5,
            current_volatility=0.02,
            temporary_impact=1e-4,
            risk_aversion=1e-6,
        )
        assert x_next > 0
        assert x_next <= 500.0

    def test_improvement_vs_vwap(self):
        solver = AlmgrenChrissSolver()
        plan = solver.solve(
            total_lots=1000.0,
            volatility=0.02,
            temporary_impact=1e-4,
            permanent_impact=5e-5,
            risk_aversion=1e-6,
            n_intervals=20,
        )
        assert plan.improvement_vs_vwap_pct >= 0 or plan.expected_cost <= plan.naive_vwap_cost

    def test_negative_lots_sell(self):
        """Negative total_lots should produce a sell schedule."""
        solver = AlmgrenChrissSolver()
        plan = solver.solve(
            total_lots=-500.0,
            volatility=0.02,
            temporary_impact=1e-4,
            permanent_impact=5e-5,
            risk_aversion=1e-6,
            n_intervals=10,
        )
        # All trades should be negative (selling)
        assert all(t <= 0 for t in plan.trade_schedule)


# ─── ExecutionMonitor tests ──────────────────────────────────────────────────


class TestExecutionMonitor:

    def _make_plan(self) -> ExecutionPlan:
        solver = AlmgrenChrissSolver()
        return solver.solve(
            total_lots=100.0,
            volatility=0.02,
            temporary_impact=1e-4,
            permanent_impact=5e-5,
            risk_aversion=1e-6,
            n_intervals=5,
        )

    def test_monitor_tracks_fills(self):
        plan = self._make_plan()
        mon = ExecutionMonitor(plan)
        mon.set_decision_price(2350.0)
        result = mon.record_fill(
            lots=plan.trade_schedule[0],
            avg_price=2350.05,
        )
        assert result["interval"] == 1
        assert result["completion_pct"] > 0

    def test_monitor_implementation_shortfall(self):
        plan = self._make_plan()
        mon = ExecutionMonitor(plan)
        mon.set_decision_price(2350.0)
        for qty in plan.trade_schedule:
            result = mon.record_fill(qty, 2350.10)
        assert "implementation_shortfall_bps" in result
        assert result["completion_pct"] == pytest.approx(100.0, abs=1.0)

    def test_total_executed_and_remaining(self):
        plan = self._make_plan()
        mon = ExecutionMonitor(plan)
        mon.record_fill(10.0, 2350.0)
        assert mon.total_executed == pytest.approx(10.0)
        assert mon.remaining == pytest.approx(plan.total_lots - 10.0)
