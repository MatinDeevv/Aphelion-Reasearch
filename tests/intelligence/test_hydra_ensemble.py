import pytest
import numpy as np

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig


def _small_config():
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=32, lstm_layers=1, attention_heads=2),
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1),
        cnn_config=CNNConfig(hidden_size=32),
        moe_config=MoEConfig(hidden_size=32),
        gate_hidden_size=32,
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
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        gate_w = out["gate_attention_weights"]
        # Shape: (batch, 1, 4) for 4 sub-models
        assert gate_w.shape == (4, 1, 4)
        # Softmax weights sum to 1
        sums = gate_w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

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
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        # Just run 1 training epoch
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
