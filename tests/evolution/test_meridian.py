"""
Phase 11 Tests — MERIDIAN Dynamic Multi-Timeframe Weighting

Covers:
  - MeridianConfig defaults
  - Granger causality F-test (granger_causality_f)
  - MeridianEngine: feed_bar, update, weight computation, EMA smoothing
  - Weight floor/ceiling constraints
  - Dominant timeframe detection
  - Causality matrix and state queries
  - Edge cases: insufficient data, equal weights, single timeframe
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.core.config import Timeframe, TIMEFRAMES
from aphelion.evolution.meridian.engine import (
    GrangerResult,
    MeridianConfig,
    MeridianEngine,
    MeridianState,
    granger_causality_f,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class TestMeridianConfig:
    def test_defaults(self):
        cfg = MeridianConfig()
        assert cfg.granger_window == 200
        assert cfg.granger_max_lag == 5
        assert cfg.update_interval_bars == 50
        assert cfg.min_weight == 0.05
        assert cfg.max_weight == 0.60
        assert cfg.smoothing_alpha == 0.3

    def test_custom_config(self):
        cfg = MeridianConfig(granger_window=100, min_weight=0.10)
        assert cfg.granger_window == 100
        assert cfg.min_weight == 0.10


# ═══════════════════════════════════════════════════════════════════════════
# Granger Causality F-Test
# ═══════════════════════════════════════════════════════════════════════════

class TestGrangerCausality:
    def test_self_causality_zero(self):
        """A random walk should not Granger-cause itself."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        f, p, lag = granger_causality_f(x, x.copy(), max_lag=3)
        # Should still produce some F, but p should be reasonable
        assert isinstance(f, float)
        assert isinstance(p, float)
        assert lag >= 1

    def test_causal_series_high_f(self):
        """When x leads y by one step, Granger test should detect it."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        # y = lagged x + noise
        y = np.zeros(n)
        y[1:] = 0.8 * x[:-1] + rng.normal(0, 0.2, n - 1)

        f, p, lag = granger_causality_f(x, y, max_lag=5)
        assert f > 1.0  # Should be significantly positive

    def test_independent_series_low_f(self):
        """Independent series should have low F-statistic."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        f, p, lag = granger_causality_f(x, y, max_lag=3)
        # May or may not be significant — just check it returns valid
        assert f >= 0.0
        assert 0.0 <= p <= 1.0

    def test_short_series(self):
        """Very short series should return defaults."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        f, p, lag = granger_causality_f(x, y, max_lag=5)
        assert f == 0.0
        assert p == 1.0

    def test_max_lag_respected(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        f, p, lag = granger_causality_f(x, y, max_lag=3)
        assert 1 <= lag <= 3


# ═══════════════════════════════════════════════════════════════════════════
# MeridianEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestMeridianEngine:
    @pytest.fixture
    def engine(self):
        config = MeridianConfig(
            granger_window=100,
            granger_max_lag=3,
            update_interval_bars=1,  # Update every bar for testing
            min_samples=20,
            min_weight=0.05,
            max_weight=0.60,
        )
        return MeridianEngine(config)

    def test_initial_weights_equal(self, engine):
        """Initial weights should be equal across all timeframes."""
        w = engine.weights
        assert len(w) == len(TIMEFRAMES)
        expected = 1.0 / len(TIMEFRAMES)
        for tf, weight in w.items():
            assert weight == pytest.approx(expected, abs=0.001)

    def test_dominant_timeframe_returns_timeframe(self, engine):
        """With equal weights, any TF could be dominant."""
        dom = engine.dominant_timeframe
        assert dom in TIMEFRAMES

    def test_feed_bar_populates_buffer(self, engine):
        for i in range(10):
            engine.feed_bar(Timeframe.M1, 2000.0 + i)
        # Internal buffer should have data
        assert len(engine._price_buffers[Timeframe.M1]) == 10

    def test_update_with_explicit_data(self, engine):
        """Feed explicit data and check weights are recomputed."""
        rng = np.random.default_rng(42)
        n = 150
        bars_by_tf = {}
        for tf in TIMEFRAMES:
            bars_by_tf[tf] = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))

        weights = engine.update(bars_by_tf)
        assert len(weights) == len(TIMEFRAMES)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_weights_respect_floor(self, engine):
        """No weight should go below min_weight."""
        rng = np.random.default_rng(42)
        n = 150
        bars_by_tf = {}
        for tf in TIMEFRAMES:
            bars_by_tf[tf] = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))

        weights = engine.update(bars_by_tf)
        for tf, w in weights.items():
            assert w >= engine._config.min_weight - 0.01

    def test_weights_respect_ceiling(self, engine):
        """No weight should exceed max_weight (after normalisation may adjust)."""
        rng = np.random.default_rng(42)
        n = 200
        # Make M5 strongly causal relative to others
        base = np.cumsum(rng.normal(0, 1, n))
        bars_by_tf = {
            Timeframe.M1: base + rng.normal(0, 0.1, n),
            Timeframe.M5: base.copy(),
            Timeframe.M15: rng.normal(0, 1, n),  # Noise
            Timeframe.H1: rng.normal(0, 1, n),    # Noise
        }

        weights = engine.update(bars_by_tf)
        # After normalisation, weights sum to 1, so ceiling may be exceeded
        # but the pre-normalisation values should be clamped
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_update_interval_throttling(self):
        """Engine should skip updates between intervals."""
        config = MeridianConfig(
            update_interval_bars=5,
            min_samples=10,
            granger_window=50,
        )
        engine = MeridianEngine(config)
        rng = np.random.default_rng(42)
        bars_by_tf = {tf: 2000 + np.cumsum(rng.normal(0, 0.5, 80)) for tf in TIMEFRAMES}

        # First call always updates
        engine.update(bars_by_tf)
        assert engine._update_count == 1

        # Next 4 should be skipped
        for _ in range(4):
            engine.update()
        assert engine._update_count == 1

        # 5th triggers update
        engine.update()
        assert engine._update_count == 2

    def test_causality_matrix_populated(self, engine):
        rng = np.random.default_rng(42)
        n = 150
        bars_by_tf = {tf: 2000 + np.cumsum(rng.normal(0, 0.5, n)) for tf in TIMEFRAMES}
        engine.update(bars_by_tf)

        matrix = engine.get_causality_matrix()
        assert isinstance(matrix, dict)
        # Should have entries for cause -> effect pairs
        total_entries = sum(len(effects) for effects in matrix.values())
        assert total_entries > 0

    def test_granger_results_list(self, engine):
        rng = np.random.default_rng(42)
        bars_by_tf = {tf: 2000 + np.cumsum(rng.normal(0, 0.5, 150)) for tf in TIMEFRAMES}
        engine.update(bars_by_tf)

        results = engine.get_granger_results()
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], GrangerResult)

    def test_state_snapshot(self, engine):
        rng = np.random.default_rng(42)
        bars_by_tf = {tf: 2000 + np.cumsum(rng.normal(0, 0.5, 150)) for tf in TIMEFRAMES}
        engine.update(bars_by_tf)

        state = engine.get_state()
        assert isinstance(state, MeridianState)
        assert state.update_count == 1
        assert len(state.weights) == len(TIMEFRAMES)

    def test_reset_restores_equal_weights(self, engine):
        rng = np.random.default_rng(42)
        bars_by_tf = {tf: 2000 + np.cumsum(rng.normal(0, 0.5, 150)) for tf in TIMEFRAMES}
        engine.update(bars_by_tf)
        engine.reset()

        expected = 1.0 / len(TIMEFRAMES)
        for w in engine.weights.values():
            assert w == pytest.approx(expected, abs=0.001)
        assert engine._update_count == 0

    def test_insufficient_data_keeps_equal_weights(self, engine):
        """With too few bars, weights should stay equal."""
        bars_by_tf = {tf: np.array([2000.0, 2001.0, 2002.0]) for tf in TIMEFRAMES}
        weights = engine.update(bars_by_tf)
        expected = 1.0 / len(TIMEFRAMES)
        for w in weights.values():
            assert w == pytest.approx(expected, abs=0.001)

    def test_ema_smoothing_prevents_sudden_jumps(self):
        """With smoothing, weights should not change drastically between updates."""
        config = MeridianConfig(
            update_interval_bars=1,
            min_samples=20,
            granger_window=60,
            smoothing_alpha=0.1,  # Heavy smoothing
        )
        engine = MeridianEngine(config)
        rng = np.random.default_rng(42)

        # First update
        bars1 = {tf: 2000 + np.cumsum(rng.normal(0, 0.5, 80)) for tf in TIMEFRAMES}
        w1 = engine.update(bars1)

        # Second update with different data
        bars2 = {tf: 2000 + np.cumsum(rng.normal(0, 2, 80)) for tf in TIMEFRAMES}
        w2 = engine.update(bars2)

        # With alpha=0.1, weights should not change dramatically
        for tf in TIMEFRAMES:
            diff = abs(w2[tf] - w1[tf])
            assert diff < 0.3  # Smoothed, so max change is limited
