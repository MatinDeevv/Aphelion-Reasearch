"""Tests for Phase 16 — AutoOptimizer (evolution/auto_optimizer.py)."""

import pytest
from datetime import datetime, timedelta, timezone

from aphelion.evolution.auto_optimizer import (
    AutoOptimizer,
    DegradationSignal,
    OptAction,
    OptimizationRun,
    PerformanceMonitor,
)


# ── PerformanceMonitor ──────────────────────────────────────────────────────

class TestPerformanceMonitor:

    def test_no_degradation_when_baseline_zero(self):
        m = PerformanceMonitor()
        signals = m.check_degradation(1.5, 0.55, 0.02)
        assert signals == []

    def test_set_baseline(self):
        m = PerformanceMonitor()
        m.set_baseline(2.0, 0.60)
        assert m._baseline_sharpe == 2.0
        assert m._baseline_wr == 0.60

    def test_sharpe_degradation_detected(self):
        m = PerformanceMonitor()
        m.set_baseline(2.0, 0.60)
        signals = m.check_degradation(1.5, 0.60, 0.02)
        sharpe_signals = [s for s in signals if s.metric == "sharpe_ratio"]
        assert len(sharpe_signals) == 1
        assert sharpe_signals[0].degradation_pct == pytest.approx(0.25, abs=0.01)

    def test_win_rate_degradation_detected(self):
        m = PerformanceMonitor()
        m.set_baseline(2.0, 0.60)
        signals = m.check_degradation(2.0, 0.50, 0.02)
        wr_signals = [s for s in signals if s.metric == "win_rate"]
        assert len(wr_signals) == 1

    def test_drawdown_breach_detected(self):
        m = PerformanceMonitor()
        signals = m.check_degradation(1.5, 0.55, 0.12)
        dd_signals = [s for s in signals if s.metric == "drawdown"]
        assert len(dd_signals) == 1

    def test_no_degradation_when_all_good(self):
        m = PerformanceMonitor()
        m.set_baseline(2.0, 0.60)
        signals = m.check_degradation(2.0, 0.60, 0.02)
        assert signals == []

    def test_multiple_signals(self):
        m = PerformanceMonitor()
        m.set_baseline(2.0, 0.60)
        signals = m.check_degradation(1.0, 0.40, 0.12)
        assert len(signals) >= 2


# ── DegradationSignal ───────────────────────────────────────────────────────

class TestDegradationSignal:

    def test_is_severe_above_20pct(self):
        sig = DegradationSignal("sharpe", 1.0, 2.0, 0.25)
        assert sig.is_severe is True

    def test_not_severe_below_20pct(self):
        sig = DegradationSignal("sharpe", 1.5, 2.0, 0.15)
        assert sig.is_severe is False


# ── AutoOptimizer ───────────────────────────────────────────────────────────

class TestAutoOptimizer:

    def test_default_state(self):
        opt = AutoOptimizer()
        assert opt.is_enabled is True
        assert opt.total_runs == 0
        assert opt.successful_runs == 0

    def test_evaluate_returns_empty_when_disabled(self):
        opt = AutoOptimizer()
        opt.disable()
        signals = opt.evaluate(0.5, 0.40, 0.12)
        assert signals == []

    def test_evaluate_returns_signals(self):
        opt = AutoOptimizer()
        opt.set_baseline(2.0, 0.60)
        signals = opt.evaluate(1.0, 0.60, 0.02)
        assert len(signals) >= 1

    def test_should_optimize_none_when_no_signals(self):
        opt = AutoOptimizer()
        assert opt.should_optimize([]) is None

    def test_should_optimize_returns_full_for_severe(self):
        opt = AutoOptimizer()
        sig = DegradationSignal("sharpe", 1.0, 2.0, 0.3)
        action = opt.should_optimize([sig])
        assert action == OptAction.FULL_REOPTIMIZE

    def test_should_optimize_forge_for_sharpe(self):
        opt = AutoOptimizer()
        sig = DegradationSignal("sharpe_ratio", 1.7, 2.0, 0.16)
        action = opt.should_optimize([sig])
        assert action == OptAction.FORGE_PARAM_OPT

    def test_should_optimize_hydra_for_win_rate(self):
        opt = AutoOptimizer()
        sig = DegradationSignal("win_rate", 0.50, 0.60, 0.16)
        action = opt.should_optimize([sig])
        assert action == OptAction.HYDRA_RETRAIN

    def test_cooldown_prevents_optimization(self):
        opt = AutoOptimizer()
        opt._last_optimization = datetime.now(timezone.utc)
        sig = DegradationSignal("sharpe", 1.0, 2.0, 0.3)
        action = opt.should_optimize([sig])
        assert action is None

    def test_run_optimization_with_callback(self):
        opt = AutoOptimizer()
        opt.register_callback(OptAction.FORGE_PARAM_OPT, lambda: {"params": "new"})
        run = opt.run_optimization(OptAction.FORGE_PARAM_OPT, "test")
        assert run.result == {"params": "new"}
        assert opt.total_runs == 1

    def test_run_optimization_no_callback(self):
        opt = AutoOptimizer()
        run = opt.run_optimization(OptAction.HYDRA_RETRAIN, "test")
        assert run.result == {"status": "no_callback"}

    def test_run_optimization_callback_failure(self):
        def bad():
            raise RuntimeError("fail")
        opt = AutoOptimizer()
        opt.register_callback(OptAction.HYDRA_RETRAIN, bad)
        run = opt.run_optimization(OptAction.HYDRA_RETRAIN, "test")
        assert run.result["status"] == "failed"

    def test_approve_and_apply(self):
        opt = AutoOptimizer()
        run = opt.run_optimization(OptAction.FEATURE_REEVAL, "test")
        opt.approve_and_apply(run)
        assert run.titan_approved is True
        assert run.applied is True
        assert opt.successful_runs == 1

    def test_enable_disable(self):
        opt = AutoOptimizer()
        opt.disable()
        assert opt.is_enabled is False
        opt.enable()
        assert opt.is_enabled is True

    def test_get_run_history(self):
        opt = AutoOptimizer()
        opt._cooldown_hours = 0  # disable cooldown for testing
        for _ in range(5):
            opt.run_optimization(OptAction.FEATURE_REEVAL, "test")
        assert len(opt.get_run_history(3)) == 3
        assert len(opt.get_run_history()) == 5
