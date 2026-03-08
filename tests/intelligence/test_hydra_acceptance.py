"""
APHELION HYDRA Phase 7 — Extended Acceptance Evidence Tests.
Covers gradient flow, checkpoint round-trip, full pipeline, and edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    DatasetConfig,
    build_dataset_from_feature_dicts,
)
from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig, HydraLSTM
from aphelion.intelligence.hydra.cnn import CNNConfig, HydraCNN
from aphelion.intelligence.hydra.moe import MoEConfig, HydraMoE
from aphelion.intelligence.hydra.trainer import (
    HydraTrainer,
    TrainerConfig,
    FocalLoss,
    QuantileLoss,
)
from aphelion.intelligence.hydra.inference import HydraInference, HydraSignal

N_CONT = len(CONTINUOUS_FEATURES)
N_CAT = len(CATEGORICAL_FEATURES)


def _small_config():
    """Tiny ensemble config for fast CPU tests."""
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=32, lstm_layers=1, attention_heads=2),
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1),
        cnn_config=CNNConfig(hidden_size=32),
        moe_config=MoEConfig(hidden_size=32),
        gate_hidden_size=32,
        dropout=0.1,
    )


def _make_batch(batch=4, seq=32):
    cont = torch.randn(batch, seq, N_CONT)
    cat = torch.randint(0, 5, (batch, seq, N_CAT))
    return cont, cat


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Flow — every sub-model receives gradients through the gate
# ═══════════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Confirm gradients propagate to all sub-models through the gate."""

    def test_all_submodels_receive_gradients(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch()
        out = model(cont, cat)
        loss = (
            out["logits_5m"].sum()
            + out["logits_15m"].sum()
            + out["logits_1h"].sum()
            + out["uncertainty"].sum()
        )
        loss.backward()

        for name in ["tft", "lstm", "cnn", "moe"]:
            sub = getattr(model, name)
            grad_params = [
                p for p in sub.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0
            ]
            assert len(grad_params) > 0, f"No gradient in {name}"

    def test_gate_projection_receives_gradients(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch()
        out = model(cont, cat)
        out["logits_1h"].sum().backward()

        # Gate projection layers should have gradients
        for proj_name in ["tft_adapter", "lstm_proj", "cnn_proj", "moe_proj"]:
            proj = getattr(model, proj_name)
            for p in proj.parameters():
                assert p.grad is not None, f"No gradient in {proj_name}"

    def test_auxiliary_loss_backward(self):
        """Auxiliary losses should backprop through each sub-model."""
        model = HydraGate(_small_config())
        cont, cat = _make_batch()
        out = model(cont, cat)

        aux_loss = sum(
            l.sum() for key in ["tft_logits", "lstm_logits", "cnn_logits", "moe_logits"]
            for l in out[key]
        )
        aux_loss.backward()

        for name in ["tft", "lstm", "cnn", "moe"]:
            sub = getattr(model, name)
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in sub.parameters()
            )
            assert has_grad, f"Aux loss doesn't reach {name}"


# ═══════════════════════════════════════════════════════════════════════════
# Loss Functions — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestLossEdgeCases:
    """Loss function correctness for edge-case inputs."""

    def test_focal_loss_all_same_class(self):
        fl = FocalLoss(gamma=2.0, alpha=[1.5, 0.5, 1.5])
        logits = torch.randn(32, 3)
        targets = torch.zeros(32, dtype=torch.long)  # All SHORT
        loss = fl(logits, targets)
        assert loss.item() > 0 and np.isfinite(loss.item())

    def test_focal_loss_no_alpha(self):
        fl = FocalLoss(gamma=2.0, alpha=None)
        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))
        loss = fl(logits, targets)
        assert loss.dim() == 0
        assert np.isfinite(loss.item())

    def test_quantile_loss_all_zeros(self):
        ql = QuantileLoss([0.1, 0.5, 0.9])
        preds = torch.zeros(8, 3)
        targets = torch.zeros(8, 3)
        loss = ql(preds, targets)
        assert loss.item() == 0.0

    def test_quantile_loss_perfect_median(self):
        ql = QuantileLoss([0.5])
        preds = torch.tensor([[1.0], [2.0], [3.0]])
        targets = torch.tensor([[1.0], [2.0], [3.0]])
        loss = ql(preds, targets)
        assert loss.item() < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint Round-Trip
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpointRoundTrip:
    """Save → Load → verify identical weights."""

    def test_save_load_preserves_weights(self, tmp_path):
        cfg = TrainerConfig(
            max_epochs=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path),
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")

        # Get state before save
        orig_state = {
            k: v.clone() for k, v in trainer.model.state_dict().items()
        }

        ckpt_path = trainer._save_checkpoint("roundtrip")
        assert ckpt_path.exists()

        # Load into fresh trainer
        trainer2 = HydraTrainer(cfg, device="cpu")
        trainer2.load_checkpoint(str(ckpt_path))

        for key in orig_state:
            assert torch.allclose(
                orig_state[key],
                trainer2.model.state_dict()[key],
            ), f"Mismatch in {key}"

    def test_checkpoint_contains_all_metadata(self, tmp_path):
        cfg = TrainerConfig(
            max_epochs=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path),
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        ckpt_path = trainer._save_checkpoint("meta")

        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required = {
            "epoch", "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "scaler_state_dict",
            "best_val_sharpe", "best_val_loss", "ensemble_config",
        }
        assert required.issubset(set(data.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# Inference Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceEdgeCases:
    """Inference with missing data, reset, and batch prediction."""

    def test_process_bar_with_nan_features(self):
        """NaN features should not crash inference (treated as 0)."""
        inf = HydraInference()
        inf._model = HydraGate(_small_config()).eval()

        # Process 64 bars, some with NaN
        for i in range(64):
            features = {f: float("nan") if i % 10 == 0 else float(np.random.randn())
                        for f in CONTINUOUS_FEATURES}
            features.update({"session": 0, "day_of_week": 0})
            result = inf.process_bar(features)

        # Should still produce a signal (NaN → 0.0 in numpy)
        # Note: NaN propagation depends on the model
        # This at least checks it doesn't crash
        assert result is None or isinstance(result, HydraSignal)

    def test_reset_resets_everything(self):
        inf = HydraInference()
        inf._buffer_idx = 64
        inf._is_primed = True
        inf._last_raw_probs = np.array([0.3, 0.4, 0.3])

        inf.reset()
        assert inf._buffer_idx == 0
        assert not inf._is_primed
        assert inf._last_raw_probs is None
        assert inf._cont_buffer.sum() == 0.0

    def test_predict_batch_shapes(self):
        inf = HydraInference()
        inf._model = HydraGate(_small_config()).eval()

        batch = 16
        cont_batch = np.random.randn(batch, 32, N_CONT).astype(np.float32)
        cat_batch = np.random.randint(0, 5, (batch, 32, N_CAT))

        signals = inf.predict_batch(cont_batch, cat_batch)
        assert len(signals) == batch
        for sig in signals:
            assert isinstance(sig, HydraSignal)
            assert sig.direction in [-1, 0, 1]
            assert 0.0 <= sig.confidence <= 1.0
            assert "TFT" in sig.gate_weights
            assert "TREND" in sig.regime_weights


# ═══════════════════════════════════════════════════════════════════════════
# HydraSignal Properties
# ═══════════════════════════════════════════════════════════════════════════

class TestHydraSignalProperties:
    """Signal dataclass edge cases."""

    def test_actionable_long_high_confidence(self):
        sig = HydraSignal(
            direction=1, confidence=0.80, uncertainty=0.1,
            probs_long=[0.8, 0.8, 0.8], probs_short=[0.1, 0.1, 0.1],
            regime_weights={"TREND": 0.5, "RANGE": 0.2, "VOL_EXP": 0.2, "NEWS": 0.1},
            gate_weights={"TFT": 0.25, "LSTM": 0.25, "CNN": 0.25, "MoE": 0.25},
        )
        assert sig.is_actionable is True
        assert sig.horizon_agreement == 1.0  # All agree LONG

    def test_actionable_short(self):
        sig = HydraSignal(
            direction=-1, confidence=0.70, uncertainty=0.2,
            probs_long=[0.1, 0.1, 0.1], probs_short=[0.7, 0.7, 0.7],
            regime_weights={"TREND": 0.25, "RANGE": 0.25, "VOL_EXP": 0.25, "NEWS": 0.25},
            gate_weights={"TFT": 0.25, "LSTM": 0.25, "CNN": 0.25, "MoE": 0.25},
        )
        assert sig.is_actionable is True
        assert sig.horizon_agreement == 1.0  # All agree SHORT

    def test_mixed_horizons_low_agreement(self):
        sig = HydraSignal(
            direction=1, confidence=0.60, uncertainty=0.15,
            probs_long=[0.7, 0.3, 0.6],  # 5m=LONG, 15m=FLAT, 1h=LONG
            probs_short=[0.1, 0.3, 0.2],
            regime_weights={"TREND": 0.25, "RANGE": 0.25, "VOL_EXP": 0.25, "NEWS": 0.25},
            gate_weights={"TFT": 0.25, "LSTM": 0.25, "CNN": 0.25, "MoE": 0.25},
        )
        # 5m=LONG, 15m=FLAT(0.4>0.3), 1h=LONG → 2/3 agreement
        assert sig.horizon_agreement >= 0.33


# ═══════════════════════════════════════════════════════════════════════════
# Full End-to-End Pipeline: Synthetic → Train → Checkpoint → Infer
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Phase 7 acceptance: complete pipeline proves ensemble works end-to-end."""

    def test_synthetic_dataset_to_inference(self, tmp_path):
        """Generate data → train 2 epochs → save → load → produce signals."""
        # 1. Build synthetic dataset
        n_bars = 300
        close_prices = np.cumsum(np.random.randn(n_bars) * 0.5) + 2000.0

        feat_dicts = []
        for i in range(n_bars):
            fd = {f: float(np.random.randn()) for f in CONTINUOUS_FEATURES}
            fd["session"] = "LONDON"
            fd["day_of_week"] = "TUE"
            fd["close"] = close_prices[i]
            feat_dicts.append(fd)

        train_ds, val_ds, test_ds, means, stds = build_dataset_from_feature_dicts(
            feat_dicts, close_prices, DatasetConfig(lookback_bars=32, batch_size=16)
        )
        assert len(train_ds) > 0

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=16, shuffle=True, drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=16, shuffle=False, drop_last=True,
        )

        # 2. Train for 2 epochs
        cfg = TrainerConfig(
            max_epochs=2,
            use_amp=False,
            checkpoint_dir=str(tmp_path),
            save_every_n_epochs=1,
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        results = trainer.train(train_loader, val_loader)
        assert results["total_epochs"] == 2
        assert results["model_params"] > 0

        # 3. Checkpoint exists
        ckpt_path = tmp_path / "hydra_ensemble_latest.pt"
        assert ckpt_path.exists()

        # 4. Load into inference
        inf = HydraInference()
        inf.load_checkpoint(str(ckpt_path))
        assert inf._model is not None

        # 5. Single sample inference
        sample = test_ds[0] if len(test_ds) > 0 else train_ds[0]
        cont_seq = sample[0].unsqueeze(0).numpy()
        cat_seq = sample[1].unsqueeze(0).numpy()
        signals = inf.predict_batch(cont_seq, cat_seq)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction in [-1, 0, 1]
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.uncertainty >= 0
        assert "TFT" in sig.gate_weights
        assert "TREND" in sig.regime_weights

    def test_loss_decreases_with_learnable_pattern(self):
        """With a simple constant-label pattern, loss should decrease."""
        n = 64
        seq = 32
        cont = torch.randn(n, seq, N_CONT)
        cat = torch.randint(0, 5, (n, seq, N_CAT))
        y = torch.full((n,), 2, dtype=torch.long)  # All LONG
        raw_ret = torch.ones(n, 3) * 0.1

        ds = torch.utils.data.TensorDataset(cont, cat, y, y, y, raw_ret)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

        cfg = TrainerConfig(
            max_epochs=5,
            use_amp=False,
            learning_rate=1e-3,
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")

        losses = []
        for _ in range(5):
            m = trainer._train_epoch(loader)
            losses.append(m["loss"])

        # Loss should decrease from first to last epoch
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"
