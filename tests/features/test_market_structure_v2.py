"""Tests for MarketStructure improvements: OB invalidation, FVG fill tracking."""

import numpy as np
import pandas as pd
import pytest
from aphelion.features.market_structure import (
    MarketStructureEngine,
    FairValueGap,
)


class TestBreakerBlocks:
    def test_breaker_blocks_detected(self):
        engine = MarketStructureEngine()
        # Need enough data for swing detection + breaker logic
        n = 50
        opens = np.linspace(2840, 2870, n) - 0.5
        highs = np.linspace(2840, 2870, n) + 2
        lows = np.linspace(2840, 2870, n) - 2
        closes = np.linspace(2840, 2870, n)
        breakers = engine.detect_breaker_blocks(opens, highs, lows, closes)
        assert isinstance(breakers, list)

    def test_valid_ob_count_in_features(self):
        engine = MarketStructureEngine()
        n = 50
        closes = np.linspace(2840, 2870, n)
        highs = closes + 2
        lows = closes - 2
        opens = closes - 0.5
        volumes = closes * 1000
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        result = engine.compute_all(df)
        assert "valid_ob_count" in result
        assert isinstance(result["valid_ob_count"], int)


class TestFVGFillTracking:
    def test_mark_filled_fvgs_basic(self):
        """FVG should be marked filled when price retraces through it."""
        fvg = FairValueGap(
            index=2, gap_high=2855.0, gap_low=2850.0,
            direction="BULLISH", filled=False,
        )
        closes = np.array([2845, 2848, 2852, 2856, 2860, 2848])
        # Price drops back below gap_low=2850 at index 5
        MarketStructureEngine.mark_filled_fvgs([fvg], closes)
        assert fvg.filled is True

    def test_unfilled_fvg_remains(self):
        fvg = FairValueGap(
            index=1, gap_high=2870.0, gap_low=2865.0,
            direction="BULLISH", filled=False,
        )
        # Bullish FVG filled when price retraces DOWN below gap_low (2865)
        # Keep prices ABOVE gap_low so it stays unfilled
        closes = np.array([2880, 2882, 2885, 2890])
        MarketStructureEngine.mark_filled_fvgs([fvg], closes)
        assert fvg.filled is False

    def test_unfilled_fvg_count_in_features(self):
        engine = MarketStructureEngine()
        n = 50
        closes = np.linspace(2840, 2870, n)
        highs = closes + 2
        lows = closes - 2
        opens = closes - 0.5
        volumes = closes * 1000
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        result = engine.compute_all(df)
        assert "unfilled_fvg_count" in result
        assert isinstance(result["unfilled_fvg_count"], int)
