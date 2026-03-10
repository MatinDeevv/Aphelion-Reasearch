"""Tests for HYDRA v2 — TCN, Transformer, XGBoost, Calibration."""

import pytest
import numpy as np

from aphelion.intelligence.hydra.calibration import (
    IsotonicCalibrator,
    DisagreementDetector,
    DynamicEnsembleWeights,
)
from aphelion.intelligence.hydra.xgb_model import (
    HydraXGBoost,
    HydraRandomForest,
    HydraTreeEnsemble,
    TreeModelConfig,
    TreePrediction,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── IsotonicCalibrator ──────────────────────────────────────────────────────

class TestIsotonicCalibrator:

    def test_uncalibrated_passthrough(self):
        cal = IsotonicCalibrator()
        assert cal.calibrate(0.7) == pytest.approx(0.7)

    def test_fit_and_calibrate(self):
        cal = IsotonicCalibrator()
        # Raw confidences and true frequencies
        raw = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        true = np.array([0.05, 0.15, 0.25, 0.55, 0.75, 0.95])
        cal.fit(raw, true)
        result = cal.calibrate(0.5)
        assert 0.0 <= result <= 1.0

    def test_fit_clamps_output(self):
        cal = IsotonicCalibrator()
        cal.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert 0.0 <= cal.calibrate(1.5) <= 1.0
        assert 0.0 <= cal.calibrate(-0.5) <= 1.0


# ── DisagreementDetector ────────────────────────────────────────────────────

class TestDisagreementDetector:

    def test_perfect_agreement(self):
        dd = DisagreementDetector()
        predictions = {"lstm": 1, "cnn": 1, "tcn": 1, "transformer": 1}
        score = dd.compute_disagreement(predictions)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_max_disagreement(self):
        dd = DisagreementDetector()
        predictions = {"a": 1, "b": -1, "c": 1, "d": -1}
        score = dd.compute_disagreement(predictions)
        assert score > 0.3

    def test_should_go_flat(self):
        dd = DisagreementDetector(high_threshold=0.3)
        predictions = {"a": 1, "b": -1, "c": 1, "d": -1}
        assert dd.should_force_flat(predictions) is True

    def test_weight_adjustment(self):
        dd = DisagreementDetector()
        adj = dd.get_weight_adjustment({"a": 1, "b": 1, "c": 1, "d": 1})
        assert adj >= 0.5


# ── DynamicEnsembleWeights ──────────────────────────────────────────────────

class TestDynamicEnsembleWeights:

    def test_default_weights(self):
        dw = DynamicEnsembleWeights(["lstm", "cnn", "tcn"])
        weights = dw.weights
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_record_and_rebalance(self):
        dw = DynamicEnsembleWeights(["a", "b"])
        for _ in range(20):
            dw.record_performance("a", 1.0)
            dw.record_performance("b", -1.0)
        dw.rebalance_from_performance()
        w = dw.weights
        assert w["a"] > w["b"]

    def test_update_weights(self):
        dw = DynamicEnsembleWeights(["a", "b"])
        dw.update_weights({"a": 0.9})
        w = dw.weights
        assert w["a"] > w["b"]


# ── HydraXGBoost ───────────────────────────────────────────────────────────

class TestHydraXGBoost:

    def test_fit_and_predict(self):
        model = HydraXGBoost()
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        pred = model.predict(X[:5])
        assert isinstance(pred, TreePrediction)
        assert pred.direction in (-1, 0, 1)
        assert 0.0 <= pred.confidence <= 1.0

    def test_feature_importance(self):
        model = HydraXGBoost()
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        pred = model.predict(X[:1])
        assert pred.feature_importance is not None
        assert len(pred.feature_importance) == 5

    def test_not_fitted_returns_zero(self):
        model = HydraXGBoost()
        pred = model.predict(np.array([[1, 2, 3]]))
        assert pred.direction == 0
        assert pred.confidence == 0.0


# ── HydraRandomForest ──────────────────────────────────────────────────────

class TestHydraRandomForest:

    def test_fit_and_predict(self):
        model = HydraRandomForest()
        X = np.random.randn(100, 8)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)
        pred = model.predict(X[:3])
        assert isinstance(pred, TreePrediction)

    def test_config_applied(self):
        cfg = TreeModelConfig(n_estimators=50, max_depth=3)
        model = HydraRandomForest(cfg)
        assert model.config.n_estimators == 50


# ── HydraTreeEnsemble ──────────────────────────────────────────────────────

class TestHydraTreeEnsemble:

    def test_ensemble_predict(self):
        ens = HydraTreeEnsemble()
        X = np.random.randn(100, 8)
        y = (X[:, 0] > 0).astype(int)
        ens.fit(X, y)
        pred = ens.predict(X[:5])
        assert isinstance(pred, TreePrediction)

    def test_ensemble_combines_models(self):
        ens = HydraTreeEnsemble()
        X = np.random.randn(200, 10)
        y = (X[:, 0] > 0).astype(int)
        ens.fit(X, y)
        pred = ens.predict(X[:10])
        assert 0.0 <= pred.confidence <= 1.0


# ── TCN + Transformer (if torch available) ─────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTCN:

    def test_tcn_forward(self):
        from aphelion.intelligence.hydra.tcn import HydraTCN, TCNConfig
        cfg = TCNConfig(hidden_size=64, num_channels=[32, 64])
        model = HydraTCN(cfg)
        cont = torch.randn(2, 32, cfg.input_size)
        cat = torch.randint(0, 5, (2, 32, 2))
        out = model(cont, cat)
        assert "latent" in out
        assert "aux_logits" in out
        assert out["latent"].shape == (2, 64)

    def test_tcn_aux_logits_shape(self):
        from aphelion.intelligence.hydra.tcn import HydraTCN, TCNConfig
        cfg = TCNConfig(hidden_size=64, num_channels=[32, 64], n_classes=3, n_horizons=3)
        model = HydraTCN(cfg)
        cont = torch.randn(4, 16, cfg.input_size)
        cat = torch.randint(0, 5, (4, 16, 2))
        out = model(cont, cat)
        assert out["aux_logits"].shape[0] == 4
        assert out["aux_logits"].shape[-1] == 3


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTransformer:

    def test_transformer_forward(self):
        from aphelion.intelligence.hydra.transformer import HydraTransformer, TransformerConfig
        cfg = TransformerConfig(hidden_size=64, n_heads=2, n_layers=2)
        model = HydraTransformer(cfg)
        cont = torch.randn(2, 32, cfg.input_size)
        cat = torch.randint(0, 5, (2, 32, 2))
        out = model(cont, cat)
        assert "latent" in out
        assert out["latent"].shape == (2, 64)

    def test_transformer_aux_logits(self):
        from aphelion.intelligence.hydra.transformer import HydraTransformer, TransformerConfig
        cfg = TransformerConfig(hidden_size=64, n_heads=2, n_layers=2, n_classes=3, n_horizons=3)
        model = HydraTransformer(cfg)
        cont = torch.randn(3, 16, cfg.input_size)
        cat = torch.randint(0, 5, (3, 16, 2))
        out = model(cont, cat)
        assert out["aux_logits"].shape[0] == 3
        assert out["aux_logits"].shape[-1] == 3
