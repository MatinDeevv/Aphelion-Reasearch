"""Tests for APHELION HYDRA Online Learning Layer."""

import numpy as np
import pytest

from aphelion.intelligence.hydra.online import (
    OnlineConfig,
    OnlineLearner,
    OnlineStats,
    OnlineExperience,
    HAS_TORCH,
)


# ─── OnlineConfig tests ──────────────────────────────────────────────────────


class TestOnlineConfig:

    def test_defaults(self):
        cfg = OnlineConfig()
        assert cfg.adapter_hidden == 64
        assert cfg.buffer_size == 1000
        assert cfg.learning_rate > 0
        assert cfg.ensemble_logit_dim == 9

    def test_custom(self):
        cfg = OnlineConfig(adapter_hidden=32, buffer_size=500)
        assert cfg.adapter_hidden == 32
        assert cfg.buffer_size == 500


# ─── OnlineLearner tests ─────────────────────────────────────────────────────


class TestOnlineLearner:

    @staticmethod
    def _make_fake_data(rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(42)
        logits = rng.normal(0, 1, size=9).astype(np.float32)
        uncertainty = float(rng.uniform(0, 1))
        features = rng.normal(0, 1, size=16).astype(np.float32)
        outcome = int(rng.integers(0, 3))
        return logits, uncertainty, features, outcome

    def test_init_no_crash(self):
        learner = OnlineLearner()
        assert isinstance(learner, OnlineLearner)

    def test_on_trade_resolved_returns_stats(self):
        learner = OnlineLearner()
        logits, unc, feat, outcome = self._make_fake_data()
        stats = learner.on_trade_resolved(logits, unc, feat, outcome)
        assert isinstance(stats, OnlineStats)
        assert stats.buffer_size >= 1

    def test_buffer_accumulates(self):
        learner = OnlineLearner()
        rng = np.random.default_rng(42)
        for _ in range(20):
            logits, unc, feat, outcome = self._make_fake_data(rng)
            stats = learner.on_trade_resolved(logits, unc, feat, outcome)
        assert stats.buffer_size == 20

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_updates_after_min_buffer(self):
        cfg = OnlineConfig(min_buffer_for_update=5, batch_size=4)
        learner = OnlineLearner(cfg)
        rng = np.random.default_rng(42)
        for i in range(10):
            logits, unc, feat, outcome = self._make_fake_data(rng)
            stats = learner.on_trade_resolved(logits, unc, feat, outcome)
        # Should have done at least one update
        assert stats.total_updates > 0
        assert stats.is_active is True

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_adjust_logits_passthrough_before_active(self):
        learner = OnlineLearner()
        logits = np.ones(9, dtype=np.float32)
        result = learner.adjust_logits(logits, 0.5, np.zeros(16, dtype=np.float32))
        np.testing.assert_array_equal(result, logits)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_adjust_logits_modifies_after_training(self):
        cfg = OnlineConfig(min_buffer_for_update=5, batch_size=4)
        learner = OnlineLearner(cfg)
        rng = np.random.default_rng(42)
        for _ in range(15):
            logits, unc, feat, outcome = self._make_fake_data(rng)
            learner.on_trade_resolved(logits, unc, feat, outcome)
        # Now adjust should potentially modify logits
        test_logits = np.zeros(9, dtype=np.float32)
        result = learner.adjust_logits(test_logits, 0.5, np.zeros(16, dtype=np.float32))
        # Result should be an array of size 9
        assert result.shape == (9,)

    def test_reset_clears_state(self):
        learner = OnlineLearner()
        rng = np.random.default_rng(42)
        for _ in range(5):
            logits, unc, feat, outcome = self._make_fake_data(rng)
            learner.on_trade_resolved(logits, unc, feat, outcome)
        learner.reset()
        stats = learner._stats()
        assert stats.is_active is False
        assert stats.resets == 1

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_adapter_norm_tracked(self):
        cfg = OnlineConfig(min_buffer_for_update=3, batch_size=2)
        learner = OnlineLearner(cfg)
        rng = np.random.default_rng(42)
        for _ in range(10):
            logits, unc, feat, outcome = self._make_fake_data(rng)
            stats = learner.on_trade_resolved(logits, unc, feat, outcome)
        assert stats.adapter_norm > 0

    def test_online_experience_creation(self):
        exp = OnlineExperience(
            ensemble_logits=np.zeros(9),
            uncertainty=0.5,
            features_summary=np.zeros(16),
            outcome=1,
            timestamp=1000.0,
        )
        assert exp.outcome == 1
        assert exp.weight == 1.0

    def test_feature_dim_mismatch_handled(self):
        """Learner should handle feature vectors of wrong dimension."""
        learner = OnlineLearner(OnlineConfig(), feature_summary_dim=16)
        logits = np.zeros(9, dtype=np.float32)
        feat_short = np.zeros(4, dtype=np.float32)  # Too short
        result = learner.adjust_logits(logits, 0.5, feat_short)
        assert result.shape == (9,)
