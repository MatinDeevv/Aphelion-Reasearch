"""Tests for FLOW module — OrderFlowAnalyzer, LiquidityZoneDetector, etc."""

import numpy as np
import pytest

from aphelion.flow.analyzer import FlowAnalyzer, FlowSignal
from aphelion.flow.liquidity import LiquidityZoneDetector, LiquidityZone
from aphelion.flow.orderflow import OrderFlowAnalyzer, OrderFlowState
from aphelion.flow.imbalance import ImbalanceTracker, ImbalanceState
from aphelion.flow.absorption import AbsorptionDetector
from aphelion.flow.sweep_detector import StopHuntDetector


class TestOrderFlowAnalyzer:
    def test_update_tick(self):
        ofa = OrderFlowAnalyzer()
        state = ofa.update_tick(2001.0, 100.0)
        assert isinstance(state, OrderFlowState)

    def test_update_tick_uptick(self):
        ofa = OrderFlowAnalyzer()
        ofa.update_tick(2000.0, 100.0)
        state = ofa.update_tick(2002.0, 150.0)
        assert state.buy_volume > 0

    def test_update_tick_downtick(self):
        ofa = OrderFlowAnalyzer()
        ofa.update_tick(2001.0, 100.0)
        state = ofa.update_tick(2000.0, 150.0)
        assert state.sell_volume > 0

    def test_bar_delta_returns_state(self):
        ofa = OrderFlowAnalyzer()
        prices = np.array([2000.0, 2001.0, 2002.0, 2003.0])
        volumes = np.array([100.0, 150.0, 120.0, 130.0])
        state = ofa.compute_bar_delta(prices, volumes)
        assert isinstance(state, OrderFlowState)

    def test_bar_delta_uptrend(self):
        ofa = OrderFlowAnalyzer()
        prices = np.array([2000.0, 2001.0, 2002.0, 2003.0])
        volumes = np.array([100.0, 150.0, 120.0, 130.0])
        state = ofa.compute_bar_delta(prices, volumes)
        assert state.delta >= 0

    def test_bar_delta_single_bar(self):
        ofa = OrderFlowAnalyzer()
        state = ofa.compute_bar_delta(np.array([2000.0]), np.array([100.0]))
        assert state.delta == 0.0

    def test_reset(self):
        ofa = OrderFlowAnalyzer()
        ofa.update_tick(2000.0, 100.0)
        ofa.reset()
        state = ofa.update_tick(2001.0, 100.0)
        assert isinstance(state, OrderFlowState)


class TestLiquidityZoneDetector:
    def test_detect_zones_returns_list(self):
        detector = LiquidityZoneDetector()
        n = 60
        base = np.linspace(2000, 2100, n)
        zones = detector.detect_zones(base + 10, base - 10, base)
        assert isinstance(zones, list)

    def test_detect_zones_with_volumes(self):
        detector = LiquidityZoneDetector()
        n = 60
        base = np.linspace(2000, 2100, n)
        zones = detector.detect_zones(base + 10, base - 10, base, np.ones(n) * 1000)
        for z in zones:
            assert isinstance(z, LiquidityZone)

    def test_detect_zones_short_data(self):
        detector = LiquidityZoneDetector()
        zones = detector.detect_zones(
            np.array([2001.0]), np.array([1999.0]), np.array([2000.0])
        )
        assert isinstance(zones, list)


class TestImbalanceTracker:
    def test_update_returns_state(self):
        tracker = ImbalanceTracker()
        state = tracker.update(buy_volume=100.0, sell_volume=80.0)
        assert isinstance(state, ImbalanceState)

    def test_zscore_builds(self):
        tracker = ImbalanceTracker()
        for i in range(60):
            state = tracker.update(buy_volume=100 + i, sell_volume=100 - i * 0.5)
        assert state.imbalance_z_score != 0.0

    def test_reset(self):
        tracker = ImbalanceTracker()
        tracker.update(100, 50)
        tracker.reset()
        assert tracker.state.imbalance_z_score == 0.0


class TestAbsorptionDetector:
    def test_update_returns_none_or_event(self):
        detector = AbsorptionDetector()
        result = detector.update(
            high=2010.0, low=1990.0, close=2005.0, volume=1000, open_price=2000.0
        )
        assert result is None or hasattr(result, 'volume_absorbed')

    def test_high_volume_absorption(self):
        detector = AbsorptionDetector()
        for i in range(55):
            detector.update(2010 + i * 0.1, 1990 + i * 0.1, 2000 + i * 0.1, 1000, 2000 + i * 0.1)
        event = detector.update(2005.6, 2005.4, 2005.5, 10000, 2005.5)
        assert event is None or event.volume_absorbed > 0


class TestStopHuntDetector:
    def test_detect_no_zones(self):
        detector = StopHuntDetector()
        result = detector.detect(
            np.array([2050.0]), np.array([2010.0]),
            np.array([2040.0]), np.array([1000.0]), zones=[],
        )
        assert result is None

    def test_detect_with_zone(self):
        detector = StopHuntDetector()
        zone = LiquidityZone(price=2050.0, zone_type="RESISTANCE", strength=2.0)
        result = detector.detect(
            highs=np.array([2045, 2046, 2055, 2048, 2040], dtype=float),
            lows=np.array([2040, 2041, 2046, 2035, 2030], dtype=float),
            closes=np.array([2043, 2044, 2047, 2037, 2032], dtype=float),
            volumes=np.array([100, 100, 500, 200, 100], dtype=float),
            zones=[zone],
        )
        assert result is None or hasattr(result, 'direction')


class TestFlowAnalyzer:
    def test_instantiation(self):
        assert FlowAnalyzer() is not None

    def test_analyze(self):
        analyzer = FlowAnalyzer()
        n = 60
        base = np.linspace(2000, 2050, n)
        signal = analyzer.analyze(base + 5, base - 5, base, np.ones(n) * 1000)
        assert isinstance(signal, FlowSignal)
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'confidence')

    def test_analyze_with_session(self):
        analyzer = FlowAnalyzer()
        n = 60
        base = np.linspace(2000, 2050, n)
        signal = analyzer.analyze(
            base + 5, base - 5, base, np.ones(n) * 1000,
            session="London", volatility_regime="RANGING",
        )
        assert signal.session == "London"
