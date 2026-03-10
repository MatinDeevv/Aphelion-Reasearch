"""Tests for Phase 20-21 — OLYMPUS & SOLA expansion modules."""

import pytest

from aphelion.governance.olympus.allocator import CapitalAllocator, Allocation
from aphelion.governance.olympus.monitor import PerformanceMonitor, HealthReport
from aphelion.governance.olympus.reporter import OlympusReporter
from aphelion.governance.council.edge_decay import EdgeDecayTracker
from aphelion.governance.council.regime_awareness import RegimeAwareness, RegimeContext
from aphelion.governance.council.improvement_loop import ImprovementLoop, ImprovementAction
from aphelion.governance.council.veto import VetoEngine, VetoResult


# ── CapitalAllocator ────────────────────────────────────────────────────────

class TestCapitalAllocator:

    def test_default_allocation(self):
        alloc = CapitalAllocator()
        a = alloc.allocation
        assert a.alpha_pct == pytest.approx(0.70)
        assert a.omega_pct == pytest.approx(0.30)

    def test_rebalance_equal_sharpe(self):
        alloc = CapitalAllocator()
        alloc.update_sharpes(1.5, 1.5)
        a = alloc.rebalance()
        assert a.alpha_pct == pytest.approx(0.5, abs=0.01)
        assert a.omega_pct == pytest.approx(0.5, abs=0.01)

    def test_rebalance_alpha_dominant(self):
        alloc = CapitalAllocator()
        alloc.update_sharpes(3.0, 1.0)
        a = alloc.rebalance()
        assert a.alpha_pct > a.omega_pct

    def test_rebalance_respects_min(self):
        alloc = CapitalAllocator()
        alloc.update_sharpes(10.0, 0.1)
        a = alloc.rebalance()
        assert a.omega_pct >= 1.0 - CapitalAllocator.MAX_PCT

    def test_custom_initial(self):
        alloc = CapitalAllocator(initial_alpha=0.50)
        assert alloc.allocation.alpha_pct == 0.50


# ── PerformanceMonitor ──────────────────────────────────────────────────────

class TestPerformanceMonitor:

    def test_evaluate_healthy(self):
        mon = PerformanceMonitor()
        report = mon.evaluate("ALPHA", 0.55, 1.2, 0.03, 2)
        assert report.is_healthy is True
        assert report.alert == "OK"

    def test_evaluate_unhealthy_win_rate(self):
        mon = PerformanceMonitor(min_win_rate=0.50)
        report = mon.evaluate("ALPHA", 0.40, 1.2, 0.03, 2)
        assert report.is_healthy is False
        assert "WR" in report.alert

    def test_evaluate_unhealthy_sharpe(self):
        mon = PerformanceMonitor(min_sharpe=1.0)
        report = mon.evaluate("ALPHA", 0.55, 0.3, 0.03, 2)
        assert report.is_healthy is False
        assert "Sharpe" in report.alert

    def test_evaluate_unhealthy_drawdown(self):
        mon = PerformanceMonitor(max_drawdown=0.05)
        report = mon.evaluate("ALPHA", 0.55, 1.2, 0.07, 2)
        assert report.is_healthy is False
        assert "DD" in report.alert

    def test_evaluate_consecutive_losses(self):
        mon = PerformanceMonitor(max_consecutive_losses=3)
        report = mon.evaluate("OMEGA", 0.55, 1.2, 0.03, 5)
        assert report.is_healthy is False

    def test_all_healthy(self):
        mon = PerformanceMonitor()
        mon.evaluate("ALPHA", 0.55, 1.2, 0.03, 2)
        mon.evaluate("OMEGA", 0.60, 1.5, 0.02, 1)
        assert mon.all_healthy is True

    def test_is_strategy_healthy_unknown(self):
        mon = PerformanceMonitor()
        assert mon.is_strategy_healthy("UNKNOWN") is True


# ── OlympusReporter ─────────────────────────────────────────────────────────

class TestOlympusReporter:

    def test_text_report_contains_fields(self):
        r = OlympusReporter()
        text = r.generate_text_report("DUAL", "RUNNING", 0.7, 0.3, 0.55, 0.60, 1.2, 1.5)
        assert "DUAL" in text
        assert "RUNNING" in text
        assert "ALPHA" in text

    def test_json_report_valid(self):
        import json
        r = OlympusReporter()
        data = json.loads(r.generate_json_report(mode="DUAL"))
        assert data["report"] == "OLYMPUS"
        assert data["mode"] == "DUAL"


# ── EdgeDecayTracker ────────────────────────────────────────────────────────

class TestEdgeDecayTracker:

    def test_no_decay_before_calibration(self):
        tracker = EdgeDecayTracker(min_trades=10)
        for _ in range(5):
            result = tracker.update(0.01)
        assert result is False
        assert tracker.decay_active is False

    def test_calibration_after_min_trades(self):
        tracker = EdgeDecayTracker(min_trades=10)
        for _ in range(20):
            tracker.update(0.01)
        assert tracker._calibrated is True

    def test_decay_detected_after_negative_shift(self):
        tracker = EdgeDecayTracker(cusum_threshold=0.5, min_trades=10)
        # Good returns
        for _ in range(15):
            tracker.update(0.02)
        # Bad returns
        for _ in range(50):
            result = tracker.update(-0.05)
        assert tracker.decay_active is True

    def test_reset(self):
        tracker = EdgeDecayTracker()
        tracker._cusum = 5.0
        tracker._decay_active = True
        tracker.reset()
        assert tracker.cusum_value == 0.0
        assert tracker.decay_active is False

    def test_trade_count(self):
        tracker = EdgeDecayTracker()
        for _ in range(10):
            tracker.update(0.01)
        assert tracker.trade_count == 10


# ── RegimeAwareness ─────────────────────────────────────────────────────────

class TestRegimeAwareness:

    def test_default_context(self):
        ra = RegimeAwareness()
        assert ra.context.regime == "UNKNOWN"

    def test_update(self):
        ra = RegimeAwareness()
        ctx = ra.update(regime="TRENDING", session="LONDON")
        assert ctx.regime == "TRENDING"
        assert ctx.session == "LONDON"

    def test_confidence_multiplier_trending(self):
        ra = RegimeAwareness()
        ra.update(regime="TRENDING")
        assert ra.confidence_multiplier() == 1.0

    def test_confidence_multiplier_crisis(self):
        ra = RegimeAwareness()
        ra.update(regime="CRISIS")
        assert ra.confidence_multiplier() == 0.5

    def test_should_tighten_volatile(self):
        ra = RegimeAwareness()
        ra.update(regime="VOLATILE")
        assert ra.should_tighten_veto() is True

    def test_should_tighten_event_block(self):
        ra = RegimeAwareness()
        ra.update(event_blocked=True)
        assert ra.should_tighten_veto() is True

    def test_no_tighten_trending(self):
        ra = RegimeAwareness()
        ra.update(regime="TRENDING")
        assert ra.should_tighten_veto() is False


# ── ImprovementLoop ─────────────────────────────────────────────────────────

class TestImprovementLoop:

    def test_empty_cycle(self):
        loop = ImprovementLoop()
        actions = loop.run_cycle()
        assert actions == []
        assert loop.cycle_count == 1

    def test_negative_contributor_flagged(self):
        loop = ImprovementLoop()
        for _ in range(10):
            loop.record_contribution("BAD_MODULE", -0.5)
        actions = loop.run_cycle()
        assert len(actions) == 1
        assert actions[0].target_module == "BAD_MODULE"
        assert actions[0].action == "REVIEW_PARAMS"

    def test_degrading_module_flagged(self):
        loop = ImprovementLoop()
        # Good history
        for _ in range(60):
            loop.record_contribution("DEGRADING", 0.5)
        # Recent bad
        for _ in range(20):
            loop.record_contribution("DEGRADING", 0.1)
        actions = loop.run_cycle()
        retrain_actions = [a for a in actions if a.action == "RETRAIN"]
        assert len(retrain_actions) >= 1

    def test_action_history_accumulated(self):
        loop = ImprovementLoop()
        for _ in range(10):
            loop.record_contribution("X", -1.0)
        loop.run_cycle()
        loop.run_cycle()
        assert len(loop.action_history) >= 2

    def test_get_module_avg(self):
        loop = ImprovementLoop()
        loop.record_contribution("MOD", 0.5)
        loop.record_contribution("MOD", 1.0)
        assert loop.get_module_avg("MOD") == pytest.approx(0.75)

    def test_get_module_avg_unknown(self):
        loop = ImprovementLoop()
        assert loop.get_module_avg("NONE") == 0.0


# ── VetoEngine ──────────────────────────────────────────────────────────────

class TestVetoEngine:

    def test_no_veto_active_mode(self):
        ve = VetoEngine()
        result = ve.evaluate("ACTIVE", False, False, False, 0.9)
        assert result.vetoed is False

    def test_veto_lockdown(self):
        ve = VetoEngine()
        result = ve.evaluate("LOCKDOWN", False, False, False, 0.9)
        assert result.vetoed is True
        assert "LOCKDOWN" in result.reason

    def test_veto_black_swan(self):
        ve = VetoEngine()
        result = ve.evaluate("ACTIVE", False, True, False, 0.9)
        assert result.vetoed is True
        assert "Black swan" in result.reason

    def test_veto_event_blocked(self):
        ve = VetoEngine()
        result = ve.evaluate("ACTIVE", False, False, True, 0.9)
        assert result.vetoed is True
        assert "Event" in result.reason

    def test_veto_edge_decay(self):
        ve = VetoEngine()
        result = ve.evaluate("ACTIVE", True, False, False, 0.9)
        assert result.vetoed is True
        assert "Edge decay" in result.reason

    def test_veto_module_unhealthy(self):
        ve = VetoEngine()
        result = ve.evaluate("ACTIVE", False, False, False, 0.9, module_healthy=False)
        assert result.vetoed is True

    def test_veto_defensive_low_confidence(self):
        ve = VetoEngine(defensive_min_confidence=0.8)
        result = ve.evaluate("DEFENSIVE", False, False, False, 0.5)
        assert result.vetoed is True

    def test_no_veto_defensive_high_confidence(self):
        ve = VetoEngine(defensive_min_confidence=0.8)
        result = ve.evaluate("DEFENSIVE", False, False, False, 0.9)
        assert result.vetoed is False

    def test_regime_tighten_raises_threshold(self):
        ve = VetoEngine(defensive_min_confidence=0.8)
        # Without tighten: 0.85 passes
        r1 = ve.evaluate("DEFENSIVE", False, False, False, 0.85)
        assert r1.vetoed is False
        # With tighten: 0.85 fails (threshold becomes 0.9)
        r2 = ve.evaluate("DEFENSIVE", False, False, False, 0.85, regime_tighten=True)
        assert r2.vetoed is True

    def test_veto_count(self):
        ve = VetoEngine()
        ve.evaluate("LOCKDOWN", False, False, False, 0.5)
        ve.evaluate("LOCKDOWN", False, False, False, 0.5)
        assert ve.veto_count == 2

    def test_history(self):
        ve = VetoEngine()
        ve.evaluate("LOCKDOWN", False, False, False, 0.5)
        assert len(ve.history) == 1
