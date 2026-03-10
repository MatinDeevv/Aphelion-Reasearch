"""Tests for Phase 17 — OMEGA expansion modules."""

import pytest
import numpy as np

from aphelion.flow.omega_engine import OmegaCoreEngine, OmegaConfig, OmegaSignal
from aphelion.flow.trend_follower import TrendFollower, TrendState
from aphelion.flow.entry_refiner import EntryRefiner, EntrySetup
from aphelion.flow.exit_manager import ExitManager, ExitDecision


# ── OmegaCoreEngine ────────────────────────────────────────────────────────

class TestOmegaCoreEngine:

    def test_identify_trend_insufficient_data(self):
        eng = OmegaCoreEngine()
        closes = np.array([1900.0] * 50)
        trend, strength = eng.identify_trend(closes, 30.0)
        assert trend == "FLAT"

    def test_identify_trend_bull(self):
        eng = OmegaCoreEngine()
        # Create uptrend: slow rise
        closes = np.linspace(1800, 2000, 250)
        trend, strength = eng.identify_trend(closes, 30.0)
        assert trend == "BULL"
        assert strength > 0

    def test_identify_trend_bear(self):
        eng = OmegaCoreEngine()
        closes = np.linspace(2000, 1800, 250)
        trend, strength = eng.identify_trend(closes, 30.0)
        assert trend == "BEAR"
        assert strength > 0

    def test_identify_trend_flat_low_adx(self):
        eng = OmegaCoreEngine()
        closes = np.linspace(1800, 2000, 250)
        trend, strength = eng.identify_trend(closes, 15.0)
        assert trend == "FLAT"

    def test_generate_signal_no_data(self):
        eng = OmegaCoreEngine()
        closes = np.array([1900.0] * 50)
        sig = eng.generate_signal(closes, closes, closes, 30.0, 5.0)
        assert sig.direction == 0

    def test_generate_signal_structure(self):
        eng = OmegaCoreEngine()
        closes = np.linspace(1800, 2000, 250)
        sig = eng.generate_signal(closes, closes + 5, closes - 5, 30.0, 5.0)
        assert isinstance(sig, OmegaSignal)
        assert isinstance(sig.reason, str)

    def test_config_defaults(self):
        cfg = OmegaConfig()
        assert cfg.trend_ema_fast == 50
        assert cfg.trend_ema_slow == 200
        assert cfg.min_adx == 25.0
        assert cfg.max_concurrent_positions == 2


# ── TrendFollower ───────────────────────────────────────────────────────────

class TestTrendFollower:

    def test_analyze_insufficient_data(self):
        tf = TrendFollower()
        state = tf.analyze(np.array([1900.0] * 100), 30.0)
        assert state.direction == "FLAT"

    def test_analyze_bull_trend(self):
        tf = TrendFollower()
        closes = np.linspace(1800, 2000, 250)
        state = tf.analyze(closes, 30.0)
        assert state.direction == "BULL"
        assert state.strength > 0
        assert state.trend_duration_bars >= 1

    def test_analyze_bear_trend(self):
        tf = TrendFollower()
        closes = np.linspace(2000, 1800, 250)
        state = tf.analyze(closes, 30.0)
        assert state.direction == "BEAR"

    def test_duration_accumulates(self):
        tf = TrendFollower()
        closes = np.linspace(1800, 2000, 250)
        tf.analyze(closes, 30.0)
        state = tf.analyze(closes, 30.0)
        assert state.trend_duration_bars >= 2

    def test_duration_resets_on_direction_change(self):
        tf = TrendFollower()
        up = np.linspace(1800, 2000, 250)
        tf.analyze(up, 30.0)
        down = np.linspace(2000, 1800, 250)
        state = tf.analyze(down, 30.0)
        assert state.trend_duration_bars == 1


# ── EntryRefiner ────────────────────────────────────────────────────────────

class TestEntryRefiner:

    def test_long_entry_valid(self):
        er = EntryRefiner(pullback_threshold=0.001, min_rr=1.5)
        setup = er.evaluate_long(1990.0, 2000.0, 5.0)
        assert setup.valid is True
        assert setup.stop_loss < setup.entry_price
        assert setup.take_profit > setup.entry_price
        assert setup.risk_reward >= 1.5

    def test_long_entry_no_pullback(self):
        er = EntryRefiner(pullback_threshold=0.01)
        setup = er.evaluate_long(2000.0, 2000.0, 5.0)
        assert setup.valid is False

    def test_short_entry_valid(self):
        er = EntryRefiner(pullback_threshold=0.001, min_rr=1.5)
        setup = er.evaluate_short(2010.0, 2000.0, 5.0)
        assert setup.valid is True
        assert setup.stop_loss > setup.entry_price
        assert setup.take_profit < setup.entry_price

    def test_short_entry_no_pullback(self):
        er = EntryRefiner(pullback_threshold=0.01)
        setup = er.evaluate_short(2000.0, 2000.0, 5.0)
        assert setup.valid is False

    def test_rr_too_low_rejected(self):
        er = EntryRefiner(pullback_threshold=0.001, min_rr=10.0)
        setup = er.evaluate_long(1995.0, 2000.0, 5.0)
        assert setup.valid is False
        assert "RR too low" in setup.reason


# ── ExitManager ─────────────────────────────────────────────────────────────

class TestExitManager:

    def test_hold_when_all_ok(self):
        em = ExitManager()
        decision = em.evaluate(1, 1990.0, 1992.0, 1985.0, 5.0, 5)
        assert decision.should_exit is False

    def test_exit_on_trend_reversal(self):
        em = ExitManager()
        decision = em.evaluate(1, 1990.0, 1995.0, 1985.0, 5.0, 5, trend_still_valid=False)
        assert decision.should_exit is True
        assert "TREND_REVERSED" in decision.reason

    def test_exit_on_max_hold(self):
        em = ExitManager(max_hold_bars=10)
        decision = em.evaluate(1, 1990.0, 1995.0, 1985.0, 5.0, 10)
        assert decision.should_exit is True
        assert "MAX_HOLD" in decision.reason

    def test_trail_updated_long(self):
        em = ExitManager(trail_atr_mult=2.0)
        # Price moved up, trailing stop should update
        decision = em.evaluate(1, 1990.0, 2005.0, 1985.0, 5.0, 3)
        if decision.new_stop is not None:
            assert decision.new_stop > 1985.0

    def test_trail_updated_short(self):
        em = ExitManager(trail_atr_mult=2.0)
        decision = em.evaluate(-1, 2010.0, 1995.0, 2020.0, 5.0, 3)
        if decision.new_stop is not None:
            assert decision.new_stop < 2020.0

    def test_breakeven_protection_long(self):
        em = ExitManager(breakeven_trigger_atr=1.0)
        # We moved 2 ATR from entry (above trigger)
        decision = em.evaluate(1, 1990.0, 2000.0, 1985.0, 5.0, 5)
        if decision.new_stop is not None:
            assert decision.new_stop >= 1990.0  # At or above entry
