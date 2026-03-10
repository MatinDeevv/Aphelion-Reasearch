"""Tests for NEMESIS expansion modules (contrarian, stress_monitor, sub-cores)."""

import pytest
from datetime import datetime, timezone

from aphelion.nemesis.contrarian import ContrarianEngine, ContrarianConfig
from aphelion.nemesis.stress_monitor import EnhancedStressMonitor, StressSnapshot
from aphelion.nemesis.chronos.core import ChronosCore
from aphelion.nemesis.leviathan.core import LeviathanCore
from aphelion.nemesis.pandora.core import PandoraCore
from aphelion.nemesis.verdict.core import VerdictCore, Verdict


# ── ContrarianEngine ────────────────────────────────────────────────────────

class TestContrarianEngine:

    def test_default_config(self):
        cfg = ContrarianConfig()
        assert cfg.cooloff_bars == 10
        assert cfg.max_consecutive == 3

    def test_evaluate_returns_signal(self):
        eng = ContrarianEngine()
        signal = eng.evaluate(
            ares_consensus=1,
            rolling_win_rate=0.6,
            regime_accuracy=0.7,
        )
        assert signal is not None

    def test_evaluate_with_losses(self):
        eng = ContrarianEngine()
        signal = eng.evaluate(
            ares_consensus=1,
            rolling_win_rate=0.3,
            regime_accuracy=0.4,
            consecutive_losses=5,
            failed_breakouts=3,
        )
        assert signal is not None

    def test_cooloff_tracking(self):
        eng = ContrarianEngine(config=ContrarianConfig(cooloff_bars=5))
        for _ in range(3):
            eng.evaluate(ares_consensus=1, rolling_win_rate=0.3, consecutive_losses=5)
        # bars_since_contrarian should be tracked

    def test_reset(self):
        eng = ContrarianEngine()
        eng.evaluate(ares_consensus=1, rolling_win_rate=0.3, consecutive_losses=5)
        eng.reset()
        assert eng._consecutive_contrarian == 0


# ── EnhancedStressMonitor ───────────────────────────────────────────────────

class TestEnhancedStressMonitor:

    def test_default_state(self):
        mon = EnhancedStressMonitor()
        assert mon.rolling_win_rate == 0.5  # default when no trades

    def test_record_trade_updates_stats(self):
        mon = EnhancedStressMonitor()
        mon.record_trade(is_win=True)
        mon.record_trade(is_win=False)
        assert mon.rolling_win_rate == pytest.approx(0.5)

    def test_high_confidence_tracking(self):
        mon = EnhancedStressMonitor()
        mon.record_trade(is_win=True, was_high_conf=True)
        mon.record_trade(is_win=False, was_high_conf=False)
        assert mon.high_conf_win_rate == 1.0

    def test_consecutive_losses(self):
        mon = EnhancedStressMonitor()
        mon.record_trade(is_win=False)
        mon.record_trade(is_win=False)
        mon.record_trade(is_win=False)
        assert mon.consecutive_losses == 3

    def test_consecutive_losses_reset_on_win(self):
        mon = EnhancedStressMonitor()
        mon.record_trade(is_win=False)
        mon.record_trade(is_win=False)
        mon.record_trade(is_win=True)
        assert mon.consecutive_losses == 0

    def test_take_snapshot(self):
        mon = EnhancedStressMonitor()
        for _ in range(10):
            mon.record_trade(is_win=True)
        snap = mon.take_snapshot()
        assert isinstance(snap, StressSnapshot)
        assert 0.0 <= snap.composite_score <= 1.0

    def test_is_stress_rising(self):
        mon = EnhancedStressMonitor()
        result = mon.is_stress_rising()
        assert isinstance(result, bool)

    def test_reset_session(self):
        mon = EnhancedStressMonitor()
        mon.record_trade(is_win=False, was_breakout=True)
        mon.reset_session()
        # reset_session clears failed_breakouts and module_failures
        assert mon._failed_breakouts == 0


# ── ChronosCore ─────────────────────────────────────────────────────────────

class TestChronosCore:

    def test_default_state(self):
        core = ChronosCore()
        assert core.recent_anomalies == []

    def test_normal_tick_no_anomaly(self):
        core = ChronosCore()
        for i in range(50):
            result = core.record_tick_rate(100.0 + i * 0.1)
        assert result is None

    def test_tick_rate_extremes_detected(self):
        core = ChronosCore(window_size=30)
        for _ in range(50):
            core.record_tick_rate(100.0)
        result = core.record_tick_rate(500.0)
        assert result is not None

    def test_volume_recording(self):
        core = ChronosCore()
        core.record_volume(10, 1000.0)
        core.record_volume(10, 1100.0)
        # Should not crash

    def test_reset(self):
        core = ChronosCore()
        core.record_tick_rate(100.0)
        core.reset()
        assert core.recent_anomalies == []


# ── LeviathanCore ───────────────────────────────────────────────────────────

class TestLeviathanCore:

    def test_default_state(self):
        core = LeviathanCore()
        assert core.is_extreme is False

    def test_normal_move_no_risk(self):
        core = LeviathanCore()
        for _ in range(50):
            core.update(price_return=0.001, volume=100, spread=2.0)
        assert core.is_extreme is False

    def test_extreme_price_move(self):
        core = LeviathanCore(sigma_threshold=3.0)
        for i in range(50):
            core.update(price_return=0.001, volume=100, spread=2.0, timestamp_idx=i)
        result = core.update(price_return=0.5, volume=100, spread=2.0, timestamp_idx=50)
        assert core.is_extreme is True

    def test_recent_events(self):
        core = LeviathanCore(sigma_threshold=3.0)
        for i in range(50):
            core.update(price_return=0.001, volume=100, spread=2.0, timestamp_idx=i)
        core.update(price_return=0.5, volume=100, spread=2.0, timestamp_idx=50)
        assert len(core.recent_events) >= 1

    def test_reset(self):
        core = LeviathanCore()
        core.update(price_return=0.001, volume=100, spread=2.0)
        core.reset()
        assert core.recent_events == []


# ── PandoraCore ─────────────────────────────────────────────────────────────

class TestPandoraCore:

    def test_default_state(self):
        core = PandoraCore()
        assert core.any_overfitting is False

    def test_no_overfit_when_gap_small(self):
        core = PandoraCore(max_sharpe_gap=1.0)
        result = core.check_sharpe_gap("test", train_sharpe=2.0, test_sharpe=1.8)
        assert result.is_overfitting is False

    def test_overfit_when_gap_large(self):
        core = PandoraCore(max_sharpe_gap=0.5)
        result = core.check_sharpe_gap("test", train_sharpe=3.0, test_sharpe=1.0)
        assert result.is_overfitting is True

    def test_win_rate_gap(self):
        core = PandoraCore(max_wr_gap=0.10)
        result = core.check_win_rate_gap("test", train_wr=0.70, test_wr=0.50)
        assert result.is_overfitting is True

    def test_full_audit(self):
        core = PandoraCore()
        results = core.full_audit("module", train_sharpe=2.0, test_sharpe=1.8,
                                  train_wr=0.60, test_wr=0.58)
        assert isinstance(results, list)
        assert len(results) >= 2

    def test_reset(self):
        core = PandoraCore(max_sharpe_gap=0.1)
        core.check_sharpe_gap("test", 3.0, 1.0)
        core.reset()
        assert core.any_overfitting is False


# ── VerdictCore ─────────────────────────────────────────────────────────────

class TestVerdictCore:

    def test_approve_when_all_clear(self):
        core = VerdictCore()
        verdict = core.evaluate(stress_score=0.1)
        assert verdict.approved is True

    def test_reject_high_stress(self):
        core = VerdictCore(stress_veto_threshold=0.8)
        verdict = core.evaluate(stress_score=0.9)
        assert verdict.approved is False

    def test_reject_temporal_anomaly(self):
        core = VerdictCore()
        verdict = core.evaluate(stress_score=0.1, temporal_anomaly=True)
        assert verdict.confidence_adjustment < 1.0

    def test_reject_tail_event(self):
        core = VerdictCore()
        verdict = core.evaluate(stress_score=0.1, tail_event=True)
        assert verdict.confidence_adjustment < 1.0

    def test_reject_overfitting(self):
        core = VerdictCore()
        verdict = core.evaluate(stress_score=0.1, overfitting_detected=True)
        assert verdict.confidence_adjustment < 1.0

    def test_confidence_adjustment_range(self):
        core = VerdictCore()
        verdict = core.evaluate(stress_score=0.3)
        assert 0.0 <= verdict.confidence_adjustment <= 1.2

    def test_stress_level_labeling(self):
        core = VerdictCore()
        low = core.evaluate(stress_score=0.1)
        assert low.stress_level == "LOW"

    def test_verdict_dataclass(self):
        core = VerdictCore()
        verdict = core.evaluate(stress_score=0.5)
        assert isinstance(verdict, Verdict)
        assert isinstance(verdict.warnings, list)
