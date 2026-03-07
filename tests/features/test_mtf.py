"""Tests for APHELION Multi-Timeframe Alignment."""

import numpy as np
import pandas as pd
from aphelion.core.config import Timeframe
from aphelion.features.mtf import MTFAlignmentEngine


class TestMTFAlignment:
    def test_all_aligned_long(self):
        engine = MTFAlignmentEngine()
        for tf in Timeframe:
            engine.update_signal(tf, "LONG")
        assert engine.alignment_count("LONG") == 4

    def test_mixed_signals(self):
        engine = MTFAlignmentEngine()
        engine.update_signal(Timeframe.M1, "LONG")
        engine.update_signal(Timeframe.M5, "LONG")
        engine.update_signal(Timeframe.M15, "SHORT")
        engine.update_signal(Timeframe.H1, "SHORT")
        assert engine.alignment_count("LONG") == 2
        assert engine.alignment_count("SHORT") == 2

    def test_weighted_alignment(self):
        engine = MTFAlignmentEngine()
        engine.update_signal(Timeframe.H1, "LONG")   # 0.30 weight
        engine.update_signal(Timeframe.M15, "LONG")   # 0.30 weight
        engine.update_signal(Timeframe.M5, "SHORT")
        engine.update_signal(Timeframe.M1, "SHORT")
        score = engine.weighted_alignment("LONG")
        assert abs(score - 0.60) < 0.01

    def test_compute_with_dataframes(self):
        engine = MTFAlignmentEngine()
        np.random.seed(42)

        # Create uptrending data
        bars = {}
        for tf in Timeframe:
            n = 30
            closes = np.linspace(2840, 2860, n)
            bars[tf] = pd.DataFrame({
                "open": closes - 0.5,
                "high": closes + 1,
                "low": closes - 1,
                "close": closes,
            })

        result = engine.compute(bars)
        assert result["mtf_dominant_direction"] == "LONG"
        assert result["mtf_alignment_count"] == 4

    def test_compute_trend_flat(self):
        engine = MTFAlignmentEngine()
        df = pd.DataFrame({
            "close": [2850.0] * 30,
            "open": [2850.0] * 30,
            "high": [2851.0] * 30,
            "low": [2849.0] * 30,
        })
        trend = engine.compute_trend(df)
        assert trend == "FLAT"
