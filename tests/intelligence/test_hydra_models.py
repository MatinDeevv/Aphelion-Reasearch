import pytest
import numpy as np

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from aphelion.intelligence.hydra.tft import TemporalFusionTransformer, TFTConfig
from aphelion.intelligence.hydra.lstm import HydraLSTM, LSTMConfig
from aphelion.intelligence.hydra.cnn import HydraCNN, CNNConfig
from aphelion.intelligence.hydra.moe import HydraMoE, MoEConfig


def _make_batch(batch_size=4, seq_len=32, n_cont=None, n_cat=None):
    n_cont = n_cont or len(CONTINUOUS_FEATURES)
    n_cat = n_cat or len(CATEGORICAL_FEATURES)
    cont = torch.randn(batch_size, seq_len, n_cont)
    cat = torch.randint(0, 5, (batch_size, seq_len, n_cat))
    return cont, cat


class TestTFT:
    def test_tft_forward_correct_output_shape(self):
        cfg = TFTConfig(hidden_dim=64, lstm_layers=1, attention_heads=2)
        model = TemporalFusionTransformer(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert out["logits_5m"].shape == (4, 3)
        assert out["probs_1h"].shape == (4, 3)
        assert out["uncertainty"].shape == (4, 1)

    def test_tft_no_nan_in_output(self):
        cfg = TFTConfig(hidden_dim=64, lstm_layers=1, attention_heads=2)
        model = TemporalFusionTransformer(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert not torch.isnan(out["logits_5m"]).any()
        assert not torch.isnan(out["logits_1h"]).any()


class TestLSTM:
    def test_lstm_forward_correct_output_shape(self):
        cfg = LSTMConfig(hidden_size=64, num_layers=1)
        model = HydraLSTM(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert out["latent"].shape == (4, 64)
        assert len(out["aux_logits"]) == 3

    def test_lstm_gradient_flows(self):
        cfg = LSTMConfig(hidden_size=64, num_layers=1)
        model = HydraLSTM(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        # FIXED: Sum ALL aux_logits so gradients flow to every aux_head
        loss = sum(logit.sum() for logit in out["aux_logits"])
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


class TestCNN:
    def test_cnn_forward_correct_output_shape(self):
        cfg = CNNConfig(hidden_size=64)
        model = HydraCNN(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert out["latent"].shape == (4, 64)
        assert len(out["aux_logits"]) == 3


class TestMoE:
    def test_moe_forward_weights_sum_to_one(self):
        cfg = MoEConfig(hidden_size=64)
        model = HydraMoE(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        weights = out["routing_weights"]
        # Softmax output should sum to ~1.0
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_moe_forward_output_shape(self):
        cfg = MoEConfig(hidden_size=64)
        model = HydraMoE(cfg)
        cont, cat = _make_batch(4, 32)
        out = model(cont, cat)
        assert out["latent"].shape == (4, 64)

    def test_model_train_eval_mode_toggle(self):
        model = HydraMoE(MoEConfig(hidden_size=64))
        model.train()
        assert model.training
        model.eval()
        assert not model.training
