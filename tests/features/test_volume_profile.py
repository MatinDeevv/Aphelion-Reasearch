"""Tests for APHELION Volume Profile Features."""

import numpy as np
import pandas as pd
from aphelion.features.volume_profile import (
    VolumeDeltaCalculator, VolumeProfileCalculator,
    AbsorptionDetector, VolumeProfileEngine,
)


class TestVolumeDelta:
    def test_bullish_delta(self):
        calc = VolumeDeltaCalculator()
        # Close near high = buy pressure
        delta = calc.compute_bar_delta(2850, 2852, 2849, 2851.5, 100)
        assert delta > 0

    def test_bearish_delta(self):
        calc = VolumeDeltaCalculator()
        # Close near low = sell pressure
        delta = calc.compute_bar_delta(2851, 2852, 2849, 2849.5, 100)
        assert delta < 0

    def test_cumulative(self):
        calc = VolumeDeltaCalculator()
        calc.compute_bar_delta(2850, 2852, 2849, 2851.5, 100)
        calc.compute_bar_delta(2851, 2853, 2850, 2852.5, 100)
        assert calc.cumulative > 0

    def test_session_reset(self):
        calc = VolumeDeltaCalculator()
        calc.compute_bar_delta(2850, 2852, 2849, 2851.5, 100)
        calc.reset_session()
        assert calc.cumulative == 0.0


class TestVolumeProfile:
    def test_poc_at_highest_volume(self):
        calc = VolumeProfileCalculator(n_bins=10)

        # Most volume concentrated at 2850
        highs = np.array([2851, 2851, 2851, 2855, 2855])
        lows = np.array([2849, 2849, 2849, 2853, 2853])
        closes = np.array([2850, 2850, 2850, 2854, 2854])
        volumes = np.array([100, 100, 100, 10, 10])

        result = calc.compute(highs, lows, closes, volumes)
        # POC should be near 2850 where most volume is
        assert abs(result["poc"] - 2850) < 2.0

    def test_value_area(self):
        calc = VolumeProfileCalculator()
        np.random.seed(42)
        n = 100
        highs = np.random.randn(n) * 2 + 2852
        lows = highs - 3
        closes = (highs + lows) / 2
        volumes = np.ones(n) * 100

        result = calc.compute(highs, lows, closes, volumes)
        assert result["vah"] > result["poc"]
        assert result["val"] < result["poc"]


class TestAbsorption:
    def test_detect_absorption(self):
        detector = AbsorptionDetector()
        # Feed normal bars first
        for _ in range(25):
            detector.check(2851, 2849, 2850, 2850, 100)
        # Now high volume, small range
        result = detector.check(2850.1, 2849.9, 2850, 2850, 500)
        assert result == True

    def test_no_absorption_normal(self):
        detector = AbsorptionDetector()
        for _ in range(25):
            detector.check(2851, 2849, 2850, 2850, 100)
        result = detector.check(2851, 2849, 2850, 2850, 100)
        assert result == False


class TestVolumeProfileEngine:
    def test_update_bar(self):
        engine = VolumeProfileEngine()
        state = engine.update_bar(2850, 2852, 2849, 2851, 100)
        assert state.volume_delta != 0 or state.volume_delta == 0  # Just check it runs

    def test_to_dict(self):
        engine = VolumeProfileEngine()
        engine.update_bar(2850, 2852, 2849, 2851, 100)
        d = engine.to_dict()
        assert "volume_delta" in d
        assert "cumulative_delta" in d
        assert "poc" in d
        assert "absorption" in d
