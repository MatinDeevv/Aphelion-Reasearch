"""Tests for APHELION Market Structure Features."""

import numpy as np
import pandas as pd
from aphelion.features.market_structure import MarketStructureEngine


class TestSwingDetection:
    def setup_method(self):
        self.engine = MarketStructureEngine(swing_confirmation=3)

    def test_detect_swing_high(self):
        # Create data with a clear peak
        highs = np.array([10, 11, 12, 15, 12, 11, 10, 9, 8])
        swings = self.engine.detect_swing_highs(highs, n=3)
        assert len(swings) == 1
        assert swings[0].price == 15
        assert swings[0].index == 3

    def test_detect_swing_low(self):
        lows = np.array([10, 9, 8, 5, 8, 9, 10, 11, 12])
        swings = self.engine.detect_swing_lows(lows, n=3)
        assert len(swings) == 1
        assert swings[0].price == 5

    def test_no_swing_in_flat(self):
        highs = np.array([10.0] * 20)
        swings = self.engine.detect_swing_highs(highs, n=3)
        assert len(swings) == 0


class TestFairValueGap:
    def setup_method(self):
        self.engine = MarketStructureEngine(fvg_min_gap_pips=0, pip_size=1.0)

    def test_bullish_fvg(self):
        # Candle 1 high=10, candle 3 low=12 → gap between 10-12
        highs = np.array([10, 11, 14])
        lows = np.array([8, 9, 12])
        fvgs = self.engine.detect_fair_value_gaps(highs, lows)
        assert len(fvgs) >= 1
        bullish = [f for f in fvgs if f.direction == "BULLISH"]
        assert len(bullish) == 1


class TestOrderBlock:
    def setup_method(self):
        self.engine = MarketStructureEngine()

    def test_bullish_ob(self):
        # Bearish candle (close < open) followed by bullish impulse (close > prior high)
        opens = np.array([2850, 2852, 2849])
        highs = np.array([2853, 2854, 2858])
        lows = np.array([2848, 2849, 2848])
        closes = np.array([2852, 2849, 2857])  # candle 1: bullish→bearish→big bullish

        obs = self.engine.detect_order_blocks(opens, highs, lows, closes)
        bullish_obs = [ob for ob in obs if ob.direction == "BULLISH"]
        # Candle at index 1 is bearish (close 2849 < open 2852)
        # Candle at index 2 closes above candle 1's high (2857 > 2854)
        assert len(bullish_obs) == 1


class TestVolumeImbalance:
    def setup_method(self):
        self.engine = MarketStructureEngine()

    def test_detect_imbalance(self):
        np.random.seed(42)
        closes = np.cumsum(np.random.randn(30)) + 2850
        opens = closes - np.random.rand(30) * 2
        volumes = np.ones(30) * 100
        volumes[-1] = 500  # Spike at end

        result = self.engine.detect_volume_imbalance(closes, opens, volumes)
        assert result[-1] == True  # Last bar should be imbalance


class TestComputeAll:
    def test_compute_all_returns_dict(self):
        engine = MarketStructureEngine(swing_confirmation=2)
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "open": np.cumsum(np.random.randn(n) * 0.5) + 2850,
            "high": np.cumsum(np.random.randn(n) * 0.5) + 2852,
            "low": np.cumsum(np.random.randn(n) * 0.5) + 2848,
            "close": np.cumsum(np.random.randn(n) * 0.5) + 2850,
            "volume": np.random.rand(n) * 100 + 50,
        })
        # Ensure high > low
        df["high"] = df[["open", "close", "high"]].max(axis=1) + 0.5
        df["low"] = df[["open", "close", "low"]].min(axis=1) - 0.5

        result = engine.compute_all(df)
        assert "swing_high_count" in result
        assert "fvg_count" in result
        assert "order_block_count" in result
        assert "last_choch_type" in result
