import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from aphelion.intelligence.hydra.strategy import HydraStrategy, StrategyConfig
from aphelion.intelligence.hydra.inference import HydraInference, HydraSignal
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar


def _make_bar(close=2850.0):
    return Bar(
        timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        timeframe=Timeframe.M1, open=close-1, high=close+2, low=close-2,
        close=close, volume=100.0, tick_volume=100, spread=0.20, is_complete=True,
    )


def _make_signal(direction=1, confidence=0.8, uncertainty=0.3):
    return HydraSignal(
        direction=direction,
        confidence=confidence,
        uncertainty=uncertainty,
        probs_long=[0.7, 0.75, 0.8],
        probs_short=[0.1, 0.1, 0.1],
        regime_weights={"TREND": 0.5, "RANGE": 0.2, "VOL_EXP": 0.2, "NEWS": 0.1},
        gate_weights={"TFT": 0.4, "LSTM": 0.2, "CNN": 0.2, "MoE": 0.2},
    )


class TestHydraStrategy:
    def test_strategy_returns_empty_when_not_primed(self):
        mock_inf = MagicMock(spec=HydraInference)
        mock_inf.process_bar.return_value = None  # Not primed
        strategy = HydraStrategy(mock_inf)
        bar = _make_bar()
        features = {"atr": 5.0}
        portfolio = Portfolio(10000.0)
        orders = strategy(bar, features, portfolio)
        assert orders == []

    def test_strategy_returns_flat_signal_no_order(self):
        mock_inf = MagicMock(spec=HydraInference)
        flat_signal = _make_signal(direction=0, confidence=0.5)
        mock_inf.process_bar.return_value = flat_signal
        strategy = HydraStrategy(mock_inf)
        bar = _make_bar()
        orders = strategy(bar, {"atr": 5.0}, Portfolio(10000.0))
        assert orders == []

    def test_strategy_signal_direction_valid(self):
        mock_inf = MagicMock(spec=HydraInference)
        mock_inf.process_bar.return_value = _make_signal(direction=1, confidence=0.8)
        strategy = HydraStrategy(mock_inf, StrategyConfig(signal_cooldown_bars=0))
        bar = _make_bar()
        orders = strategy(bar, {"atr": 5.0}, Portfolio(10000.0))
        assert len(orders) == 1
        # Should be a BUY order for LONG signal
        from aphelion.backtest.order import OrderSide
        assert orders[0].side == OrderSide.BUY

    def test_strategy_confidence_in_range(self):
        signal = _make_signal(direction=1, confidence=0.8)
        assert 0.0 <= signal.confidence <= 1.0

    def test_strategy_low_confidence_no_order(self):
        mock_inf = MagicMock(spec=HydraInference)
        mock_inf.process_bar.return_value = _make_signal(direction=1, confidence=0.3)
        strategy = HydraStrategy(mock_inf, StrategyConfig(min_confidence=0.55))
        orders = strategy(_make_bar(), {"atr": 5.0}, Portfolio(10000.0))
        assert orders == []

    def test_strategy_high_uncertainty_no_order(self):
        mock_inf = MagicMock(spec=HydraInference)
        mock_inf.process_bar.return_value = _make_signal(direction=1, confidence=0.9, uncertainty=0.9)
        strategy = HydraStrategy(mock_inf, StrategyConfig(
            uncertainty_ceiling=0.8, signal_cooldown_bars=0,
        ))
        orders = strategy(_make_bar(), {"atr": 5.0}, Portfolio(10000.0))
        assert orders == []
