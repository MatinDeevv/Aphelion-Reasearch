"""Tests for APHELION Data Layer."""

import asyncio
import time
import pytest
from aphelion.core.data_layer import (
    Tick, Bar, BarAggregator, DataQualityValidator, DataLayer,
)
from aphelion.core.event_bus import EventBus
from aphelion.core.config import Timeframe


class TestTick:
    def test_mid_price(self):
        tick = Tick(timestamp=time.time(), bid=2850.0, ask=2850.50, last=2850.25, volume=1.0)
        assert tick.mid == 2850.25

    def test_spread(self):
        tick = Tick(timestamp=time.time(), bid=2850.0, ask=2850.50, last=2850.25, volume=1.0)
        assert tick.spread == 0.5


class TestBarAggregator:
    def test_first_tick_creates_bar(self):
        agg = BarAggregator(Timeframe.M1)
        tick = Tick(timestamp=1709568000.0, bid=2850.0, ask=2850.50, last=2850.25, volume=1.0)
        result = agg.process_tick(tick)
        assert result is None  # No completed bar yet

    def test_new_bar_on_boundary(self):
        agg = BarAggregator(Timeframe.M1)

        # First tick at 10:00:30
        t1 = Tick(timestamp=1709568030.0, bid=2850.0, ask=2850.50, last=2850.25, volume=1.0)
        agg.process_tick(t1)

        # Second tick at 10:01:05 (crosses minute boundary)
        t2 = Tick(timestamp=1709568065.0, bid=2851.0, ask=2851.50, last=2851.25, volume=2.0)
        completed = agg.process_tick(t2)

        assert completed is not None
        assert completed.is_complete
        assert completed.open == 2850.25
        assert completed.close == 2850.25
        assert completed.volume == 1.0

    def test_ohlcv_tracking(self):
        agg = BarAggregator(Timeframe.M1)
        base_time = 1709568000.0

        ticks = [
            Tick(base_time + 1, 2850.0, 2850.50, 2850.25, 1.0),
            Tick(base_time + 2, 2851.0, 2851.50, 2851.25, 2.0),  # New high
            Tick(base_time + 3, 2849.0, 2849.50, 2849.25, 1.5),  # New low
            Tick(base_time + 4, 2850.5, 2851.0, 2850.75, 1.0),
        ]

        for tick in ticks:
            agg.process_tick(tick)

        # Force bar close
        close_tick = Tick(base_time + 61, 2850.0, 2850.50, 2850.25, 1.0)
        bar = agg.process_tick(close_tick)

        assert bar is not None
        assert bar.open == 2850.25
        assert bar.high == 2851.25
        assert bar.low == 2849.25
        assert bar.close == 2850.75
        assert bar.tick_volume == 4
        assert bar.volume == 5.5


class TestDataQualityValidator:
    def setup_method(self):
        self.validator = DataQualityValidator()

    def test_valid_tick(self):
        tick = Tick(time.time(), 2850.0, 2850.50, 2850.25, 1.0)
        valid, error = self.validator.validate_tick(tick)
        assert valid is True
        assert error is None

    def test_negative_price(self):
        tick = Tick(time.time(), -1.0, 2850.50, 2850.25, 1.0)
        valid, error = self.validator.validate_tick(tick)
        assert valid is False
        assert "Non-positive" in error

    def test_crossed_spread(self):
        tick = Tick(time.time(), 2851.0, 2850.0, 2850.25, 1.0)
        valid, error = self.validator.validate_tick(tick)
        assert valid is False
        assert "crossed" in error.lower()

    def test_wide_spread(self):
        tick = Tick(time.time(), 2850.0, 2910.0, 2850.25, 1.0)
        valid, error = self.validator.validate_tick(tick)
        assert valid is False
        assert "wide" in error.lower()

    def test_valid_bar(self):
        bar = Bar(
            timestamp=None, timeframe=Timeframe.M1,
            open=2850.25, high=2851.0, low=2849.5, close=2850.75,
            volume=100.0, tick_volume=50, spread=0.5,
        )
        valid, error = self.validator.validate_bar(bar)
        assert valid is True

    def test_invalid_bar_high_low(self):
        bar = Bar(
            timestamp=None, timeframe=Timeframe.M1,
            open=2850.25, high=2849.0, low=2851.0, close=2850.75,
            volume=100.0, tick_volume=50, spread=0.5,
        )
        valid, error = self.validator.validate_bar(bar)
        assert valid is False


class TestDataLayer:
    @pytest.fixture
    def data_layer(self):
        bus = EventBus()
        return DataLayer(bus)

    @pytest.mark.asyncio
    async def test_process_valid_tick(self, data_layer):
        tick = Tick(time.time(), 2850.0, 2850.50, 2850.25, 1.0)
        await data_layer.process_tick(tick)
        assert data_layer.stats["tick_count"] == 1

    @pytest.mark.asyncio
    async def test_process_invalid_tick(self, data_layer):
        tick = Tick(time.time(), -1.0, 2850.50, 2850.25, 1.0)
        await data_layer.process_tick(tick)
        assert data_layer.stats["tick_count"] == 0

    def test_get_empty_bars(self, data_layer):
        bars = data_layer.get_bars(Timeframe.M1)
        assert bars == []

    def test_get_empty_bars_df(self, data_layer):
        df = data_layer.get_bars_df(Timeframe.M1)
        assert len(df) == 0
