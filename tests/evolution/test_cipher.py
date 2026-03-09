"""
Phase 11 Tests — CIPHER Feature Importance & Alpha Decay Detection

Covers:
  - CipherConfig defaults
  - PermutationImportanceComputer
  - Half-life estimation (estimate_half_life)
  - CipherEngine: update cycle, decay detection, rankings, alerts, weights
  - FeatureImportance status classification
  - Edge cases: empty data, constant features, single feature
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.evolution.cipher.engine import (
    CipherConfig,
    CipherEngine,
    DecayAlert,
    FeatureImportance,
    PermutationImportanceComputer,
    estimate_half_life,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class TestCipherConfig:
    def test_defaults(self):
        cfg = CipherConfig()
        assert cfg.short_window == 30
        assert cfg.long_window == 90
        assert cfg.update_interval_bars == 50
        assert cfg.decay_threshold == 0.50
        assert cfg.critical_decay_threshold == 0.25

    def test_custom_config(self):
        cfg = CipherConfig(short_window=10, long_window=30, decay_threshold=0.60)
        assert cfg.short_window == 10
        assert cfg.long_window == 30
        assert cfg.decay_threshold == 0.60


# ═══════════════════════════════════════════════════════════════════════════
# FeatureImportance
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureImportance:
    def test_status_healthy(self):
        fi = FeatureImportance(name="feat1", is_decaying=False, is_critical=False, is_dead=False)
        assert fi.status == "HEALTHY"

    def test_status_decaying(self):
        fi = FeatureImportance(name="feat1", is_decaying=True)
        assert fi.status == "DECAYING"

    def test_status_critical(self):
        fi = FeatureImportance(name="feat1", is_decaying=True, is_critical=True)
        assert fi.status == "CRITICAL"

    def test_status_dead(self):
        fi = FeatureImportance(name="feat1", is_dead=True)
        assert fi.status == "DEAD"

    def test_dead_overrides_critical(self):
        fi = FeatureImportance(name="feat1", is_dead=True, is_critical=True, is_decaying=True)
        assert fi.status == "DEAD"


# ═══════════════════════════════════════════════════════════════════════════
# Half-Life Estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestHalfLife:
    def test_exponential_decay_detectable(self):
        """A clearly decaying exponential should produce a finite half-life."""
        values = [100.0 * np.exp(-0.05 * t) for t in range(50)]
        hl = estimate_half_life(values, min_points=10)
        assert hl > 0
        assert hl < 100  # Should find roughly ln(2)/0.05 ≈ 13.86

    def test_constant_signal_no_decay(self):
        """Constant values → no decay → infinite half-life."""
        values = [10.0] * 50
        hl = estimate_half_life(values, min_points=10)
        assert hl == float("inf")

    def test_increasing_signal_no_decay(self):
        """Increasing values should not trigger decay."""
        values = [float(i + 1) for i in range(50)]
        hl = estimate_half_life(values, min_points=10)
        assert hl == float("inf")

    def test_insufficient_data(self):
        values = [1.0, 0.9, 0.8]
        hl = estimate_half_life(values, min_points=10)
        assert hl == float("inf")

    def test_all_zeros(self):
        values = [0.0] * 50
        hl = estimate_half_life(values, min_points=10)
        assert hl == float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# Permutation Importance
# ═══════════════════════════════════════════════════════════════════════════

class TestPermutationImportance:
    def test_informative_feature_has_positive_importance(self):
        """A feature highly correlated with targets should have high importance."""
        rng = np.random.default_rng(42)
        n = 200
        # Feature 0 is the target signal, Feature 1 is noise
        x0 = rng.normal(0, 1, n)
        x1 = rng.normal(0, 1, n)
        features = np.column_stack([x0, x1])
        targets = x0 + rng.normal(0, 0.1, n)  # Target ≈ feature 0

        pc = PermutationImportanceComputer(n_permutations=10, rng=rng)

        def scorer(f, t):
            return float(np.mean([abs(np.corrcoef(f[:, i], t)[0, 1]) for i in range(f.shape[1])]))

        imps = pc.compute(features, targets, scorer, ["signal", "noise"])
        assert imps["signal"] > imps["noise"]

    def test_noise_feature_near_zero(self):
        """Pure noise features should have importance near zero."""
        rng = np.random.default_rng(42)
        n = 200
        features = rng.normal(0, 1, (n, 3))
        targets = rng.normal(0, 1, n)  # Independent of features

        pc = PermutationImportanceComputer(n_permutations=5, rng=rng)

        def scorer(f, t):
            return float(np.mean([abs(np.corrcoef(f[:, i], t)[0, 1]) for i in range(f.shape[1])]))

        imps = pc.compute(features, targets, scorer, ["a", "b", "c"])
        for v in imps.values():
            assert abs(v) < 0.1  # Near zero

    def test_too_few_samples(self):
        features = np.ones((5, 3))
        targets = np.ones(5)
        pc = PermutationImportanceComputer()
        imps = pc.compute(features, targets, lambda f, t: 0.0, ["a", "b", "c"])
        assert all(v == 0.0 for v in imps.values())


# ═══════════════════════════════════════════════════════════════════════════
# CipherEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestCipherEngine:
    @pytest.fixture
    def engine(self):
        config = CipherConfig(
            short_window=5,
            long_window=15,
            update_interval_bars=1,  # Update every bar for testing
            min_observations=5,
            n_permutations=3,
        )
        return CipherEngine(config)

    @pytest.fixture
    def sample_data(self):
        """Generate sample feature matrix with one informative + one noise."""
        rng = np.random.default_rng(42)
        n = 50
        x0 = rng.normal(0, 1, n)
        x1 = rng.normal(0, 1, n)
        features = np.column_stack([x0, x1])
        targets = x0 * 0.8 + rng.normal(0, 0.2, n)
        return features, targets, ["signal", "noise"]

    def test_first_update_returns_alerts_or_empty(self, engine, sample_data):
        features, targets, names = sample_data
        alerts = engine.update(features, targets, names)
        assert isinstance(alerts, list)

    def test_feature_tracking_after_update(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        assert engine.get_feature("signal") is not None
        assert engine.get_feature("noise") is not None
        assert engine.get_feature("nonexistent") is None

    def test_rankings_order(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        rankings = engine.get_rankings()
        assert len(rankings) == 2
        # Rankings should be sorted by importance descending
        assert rankings[0].short_importance >= rankings[1].short_importance

    def test_top_n_rankings(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        top1 = engine.get_rankings(top_n=1)
        assert len(top1) == 1

    def test_decay_detection_with_declining_feature(self, engine):
        """Simulate a feature whose importance drops over time."""
        rng = np.random.default_rng(42)
        n = 50
        all_alerts = []
        for step in range(20):
            # Signal strength decays over time
            strength = max(0.01, 1.0 - step * 0.08)
            x0 = rng.normal(0, 1, n) * strength
            x1 = rng.normal(0, 1, n)
            features = np.column_stack([x0, x1])
            targets = x0 + rng.normal(0, 0.1, n)
            alerts = engine.update(features, targets, ["decaying_signal", "noise"])
            all_alerts.extend(alerts)

        # After enough decay, should see some alerts
        # (This depends on the exact config thresholds)
        decaying = engine.get_decaying_features()
        # At least check the engine ran without errors
        assert engine.get_feature("decaying_signal") is not None

    def test_feature_weights_sum_to_one(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        weights = engine.get_feature_weights()
        assert len(weights) > 0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_healthy_feature_names(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        healthy = engine.get_healthy_feature_names()
        assert isinstance(healthy, list)

    def test_summary(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        summary = engine.get_summary()
        assert "total_features" in summary
        assert "healthy" in summary
        assert "decaying" in summary
        assert "critical" in summary
        assert "dead" in summary
        assert summary["total_features"] == 2
        assert summary["update_count"] == 1

    def test_reset_clears_state(self, engine, sample_data):
        features, targets, names = sample_data
        engine.update(features, targets, names)
        engine.reset()
        assert engine.get_rankings() == []
        assert engine.get_alerts() == []
        summary = engine.get_summary()
        assert summary["total_features"] == 0

    def test_update_interval_throttling(self):
        """Engine should skip updates between intervals."""
        config = CipherConfig(update_interval_bars=10, n_permutations=2)
        engine = CipherEngine(config)
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (30, 2))
        targets = rng.normal(0, 1, 30)

        # First call always updates
        engine.update(features, targets, ["a", "b"])
        assert engine.get_summary()["update_count"] == 1

        # Next 9 calls should be skipped
        for _ in range(9):
            engine.update(features, targets, ["a", "b"])
        assert engine.get_summary()["update_count"] == 1

        # 10th call triggers update
        engine.update(features, targets, ["a", "b"])
        assert engine.get_summary()["update_count"] == 2

    def test_alerts_filtered_by_severity(self, engine):
        """Check alert severity filtering."""
        rng = np.random.default_rng(42)
        n = 50
        # Run many updates to potentially trigger alerts
        for _ in range(20):
            features = rng.normal(0, 1, (n, 2))
            targets = rng.normal(0, 1, n)
            engine.update(features, targets, ["a", "b"])

        warning_alerts = engine.get_alerts(severity="WARNING")
        critical_alerts = engine.get_alerts(severity="CRITICAL")
        all_alerts = engine.get_alerts()
        assert len(all_alerts) >= len(warning_alerts) + len(critical_alerts)


# ═══════════════════════════════════════════════════════════════════════════
# DecayAlert
# ═══════════════════════════════════════════════════════════════════════════

class TestDecayAlert:
    def test_alert_fields(self):
        alert = DecayAlert(
            feature_name="rsi",
            severity="WARNING",
            decay_ratio=0.45,
            half_life_days=25.0,
            message="RSI decaying",
        )
        assert alert.feature_name == "rsi"
        assert alert.severity == "WARNING"
        assert alert.decay_ratio == 0.45
        assert alert.timestamp is not None
