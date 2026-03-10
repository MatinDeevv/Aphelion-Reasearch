import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock
from datetime import datetime, timezone

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from aphelion.intelligence.hydra.adversarial import (
    AdversarialFeaturePerturbationDetector,
    AdversarialConfig,
)
from aphelion.intelligence.hydra.inference import HydraSignal, HydraInference
from aphelion.intelligence.hydra.strategy import HydraStrategy, StrategyConfig
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar


class _TinyHydraModel(torch.nn.Module):
    def __init__(self, n_cont: int):
        super().__init__()
        self.fc = torch.nn.Linear(n_cont, 3)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            self.fc.weight[2, 0] = 3.0
            self.fc.weight[0, 0] = -2.5

    def forward(self, cont_inputs, cat_inputs):
        x = cont_inputs[:, -1, :]
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        b = logits.shape[0]
        return {
            "logits_1h": logits,
            "probs_1h": probs,
            "probs_5m": probs,
            "probs_15m": probs,
            "uncertainty": torch.ones((b, 1), device=logits.device) * 0.2,
            "gate_attention_weights": torch.softmax(torch.randn((b, 1, 6), device=logits.device), dim=-1),
            "moe_routing_weights": torch.softmax(torch.randn((b, 4), device=logits.device), dim=-1),
        }


def _make_bar(close=2850.0):
    return Bar(
        timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        timeframe=Timeframe.M1,
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=100.0,
        tick_volume=100,
        spread=0.20,
        is_complete=True,
    )


def test_adversarial_detector_returns_bounded_scores():
    n_cont = len(CONTINUOUS_FEATURES)
    n_cat = len(CATEGORICAL_FEATURES)

    model = _TinyHydraModel(n_cont)
    detector = AdversarialFeaturePerturbationDetector(
        AdversarialConfig(enabled=True, epsilon=0.2, step_size=0.05, steps=4)
    )

    cont = torch.zeros((1, 64, n_cont), dtype=torch.float32)
    cont[:, -1, 0] = 0.2
    cat = torch.zeros((1, 64, n_cat), dtype=torch.long)

    assessment = detector.assess(model, cont, cat)

    assert 0.0 <= assessment.risk_score <= 1.0
    assert assessment.raw_confidence >= assessment.robust_confidence
    assert assessment.boundary_distance >= 0.0


def test_hydra_strategy_blocks_high_adversarial_risk():
    mock_inf = MagicMock(spec=HydraInference)
    mock_inf.process_bar.return_value = HydraSignal(
        direction=1,
        confidence=0.85,
        uncertainty=0.2,
        probs_long=[0.8, 0.8, 0.85],
        probs_short=[0.05, 0.05, 0.05],
        regime_weights={"TREND": 0.6, "RANGE": 0.1, "VOL_EXP": 0.2, "NEWS": 0.1},
        gate_weights={"TFT": 0.2, "LSTM": 0.2, "CNN": 0.2, "MoE": 0.2, "TCN": 0.1, "Transformer": 0.1},
        adversarial_risk=0.95,
    )

    strategy = HydraStrategy(
        mock_inf,
        StrategyConfig(signal_cooldown_bars=0, max_adversarial_risk=0.8),
    )

    orders = strategy(_make_bar(), {"atr": 5.0}, Portfolio(10_000.0))
    assert orders == []
