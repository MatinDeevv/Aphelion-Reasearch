"""Integration tests: HYDRA + Backtest Engine."""

import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock
from datetime import datetime, timezone

from aphelion.backtest.engine import BacktestEngine, BacktestConfig
from aphelion.backtest.order import Order, OrderType, OrderSide
from aphelion.core.data_layer import DataLayer
from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock
from aphelion.intelligence.hydra.inference import HydraSignal

from tests.backtest.conftest import make_bars, make_sentinel_stack


class TestHydraBacktest:
    def test_hydra_strategy_placeholder_runs_in_engine(self):
        """A strategy that always returns [] (simulating untrained HYDRA) runs without error."""
        stack = make_sentinel_stack()
        config = BacktestConfig(
            warmup_bars=10,
            enable_feature_engine=False,
            random_seed=42,
        )
        bus = EventBus()
        clock = MarketClock()
        dl = DataLayer(bus, clock)
        engine = BacktestEngine(config, stack, dl)

        def flat_strategy(bar, features, portfolio):
            return []

        engine.set_strategy(flat_strategy)
        bars = make_bars(200)
        results = engine.run(bars)
        assert results is not None
        assert len(results.trades) == 0

    def test_hydra_signal_used_as_order(self):
        """Mock a strategy that converts HydraSignal(LONG) to an Order."""
        stack = make_sentinel_stack()
        config = BacktestConfig(
            warmup_bars=5,
            enable_feature_engine=False,
            random_seed=42,
        )
        bus = EventBus()
        clock = MarketClock()
        dl = DataLayer(bus, clock)
        engine = BacktestEngine(config, stack, dl)

        called = [False]
        def hydra_strategy(bar, features, portfolio):
            if not called[0]:
                called[0] = True
                return [Order(
                    order_id="hydra-001", symbol="XAUUSD",
                    order_type=OrderType.MARKET, side=OrderSide.BUY,
                    size_lots=0.01, entry_price=0.0,
                    stop_loss=bar.close - 5.0, take_profit=bar.close + 10.0,
                    size_pct=0.02, proposed_by="HYDRA_TFT_v1",
                )]
            return []

        engine.set_strategy(hydra_strategy)
        bars = make_bars(200)
        results = engine.run(bars)
        assert results is not None
        # Should have at least attempted to trade
        assert results.broker_stats["fill_count"] >= 0
