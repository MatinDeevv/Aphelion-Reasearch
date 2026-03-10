import pytest
import numpy as np

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig
from aphelion.intelligence.hydra.tcn import TCNConfig
from aphelion.intelligence.hydra.transformer import TransformerConfig


def _small_config():
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=32, lstm_layers=1, attention_heads=2),
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1, n_attention_heads=2),
        cnn_config=CNNConfig(hidden_size=32, channels=(16, 32, 64, 128)),
        moe_config=MoEConfig(hidden_size=32, expert_hidden_size=48, num_experts=4, top_k=2),
        tcn_config=TCNConfig(hidden_size=32, num_channels=[16, 32, 32]),
        transformer_config=TransformerConfig(hidden_size=32, n_heads=2, n_layers=2, dim_feedforward=64),
        gate_hidden_size=32,
        gate_n_heads=2,
        gate_n_interaction_layers=1,
        model_dropout=0.0,       # Disable for deterministic tests
        dropout=0.1,
    )


def _make_batch(batch_size=4, seq_len=32):
    n_cont = len(CONTINUOUS_FEATURES)
    n_cat = len(CATEGORICAL_FEATURES)
    cont = torch.randn(batch_size, seq_len, n_cont)
    cat = torch.randint(0, 5, (batch_size, seq_len, n_cat))
    return cont, cat


class TestHydraEnsemble:
    def test_ensemble_forward_output_shape(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert out["logits_5m"].shape == (4, 3)
        assert out["logits_15m"].shape == (4, 3)
        assert out["logits_1h"].shape == (4, 3)
        assert out["uncertainty"].shape == (4, 1)

    def test_ensemble_no_nan_output(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        for key in ["logits_5m", "logits_15m", "logits_1h"]:
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_ensemble_deterministic_in_eval_mode(self):
        model = HydraGate(_small_config())
        model.eval()
        cont, cat = _make_batch(4, 32)
        with torch.no_grad():
            out1 = model(cont, cat)
            out2 = model(cont, cat)
        assert torch.allclose(out1["logits_1h"], out2["logits_1h"])

    def test_ensemble_gate_attention_weights(self):
        model = HydraGate(_small_config())
        model.eval()  # Disable dropout for clean attention sums
        cont, cat = _make_batch(4, 32)
        with torch.no_grad():
            out = model(cont, cat)
        gate_w = out["gate_attention_weights"]
        # Shape: (batch, n_queries, n_models) — 4 queries (3 horizons + global), 6 sub-models
        assert gate_w.shape == (4, 4, 6)
        # Attention weights sum to 1 over models dimension
        sums = gate_w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_ensemble_confidence_output(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert "confidence" in out
        assert out["confidence"].shape == (4, 1)
        # Confidence should be in [0, 1] (sigmoid)
        assert (out["confidence"] >= 0).all()
        assert (out["confidence"] <= 1).all()

    def test_ensemble_moe_load_balance_loss(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert "moe_load_balance_loss" in out
        assert out["moe_load_balance_loss"].ndim == 0  # scalar

    def test_ensemble_all_aux_logits_present(self):
        model = HydraGate(_small_config())
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        for key in ["tft_logits", "lstm_logits", "cnn_logits", "moe_logits",
                     "tcn_logits", "transformer_logits"]:
            assert key in out, f"Missing aux logits: {key}"
            logits = out[key]
            assert len(logits) == 3, f"{key} should have 3 horizon logits"
            for l in logits:
                assert l.shape == (4, 3), f"{key} logits wrong shape"

    def test_trainer_runs_one_epoch_no_exception(self):
        from torch.utils.data import TensorDataset, DataLoader
        from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig

        n_cont = len(CONTINUOUS_FEATURES)
        n_cat = len(CATEGORICAL_FEATURES)
        n_samples = 32
        seq_len = 32

        cont = torch.randn(n_samples, seq_len, n_cont)
        cat = torch.randint(0, 5, (n_samples, seq_len, n_cat))
        y5m = torch.randint(0, 3, (n_samples,))
        y15m = torch.randint(0, 3, (n_samples,))
        y1h = torch.randint(0, 3, (n_samples,))
        raw_ret = torch.randn(n_samples, 3)

        ds = TensorDataset(cont, cat, y5m, y15m, y1h, raw_ret)
        loader = DataLoader(ds, batch_size=8, shuffle=True)

        cfg = TrainerConfig(
            max_epochs=1,
            use_amp=False,
            warmup_epochs=0,
            gradient_accumulation_steps=1,  # No accum for fast test
            mixup_alpha=0.0,                # No mixup for fast test
            swa_start_epoch=999,            # Disable SWA for test
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        trainer._model.train()
        metrics = trainer._train_epoch(loader)
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])

    def test_trainer_loss_decreases_over_epochs(self):
        from torch.utils.data import TensorDataset, DataLoader
        from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig

        n_cont = len(CONTINUOUS_FEATURES)
        n_cat = len(CATEGORICAL_FEATURES)
        n_samples = 64
        seq_len = 32

        # Create simple learnable pattern
        cont = torch.randn(n_samples, seq_len, n_cont)
        cat = torch.randint(0, 5, (n_samples, seq_len, n_cat))
        # All labels are class 2 (LONG) - easy pattern
        y5m = torch.full((n_samples,), 2, dtype=torch.long)
        y15m = torch.full((n_samples,), 2, dtype=torch.long)
        y1h = torch.full((n_samples,), 2, dtype=torch.long)
        raw_ret = torch.ones(n_samples, 3) * 0.1

        ds = TensorDataset(cont, cat, y5m, y15m, y1h, raw_ret)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        cfg = TrainerConfig(
            max_epochs=10,
            use_amp=False,
            learning_rate=1e-3,
            warmup_epochs=0,
            gradient_accumulation_steps=1,
            mixup_alpha=0.0,
            swa_start_epoch=999,
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")

        losses = []
        for _ in range(10):
            metrics = trainer._train_epoch(loader)
            losses.append(metrics["loss"])
        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_inference_returns_signal(self):
        from aphelion.intelligence.hydra.inference import HydraInference, HydraSignal
        # Can't test with actual checkpoint, but test that process_bar returns None
        # when buffer is not primed
        inf = HydraInference(checkpoint_path=None, device="cpu")
        features = {f: 0.0 for f in CONTINUOUS_FEATURES}
        features.update({f: 0 for f in CATEGORICAL_FEATURES})
        result = inf.process_bar(features)
        # First call: buffer not primed yet (needs 64 bars)
        assert result is None
