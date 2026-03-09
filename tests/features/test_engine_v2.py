"""Tests for FeatureEngine improvements: Wilder RSI, NaN validation."""

import numpy as np
import pytest
from aphelion.features.engine import FeatureEngine


class TestWilderRSI:
    def test_rsi_bounded_0_to_100(self):
        engine = FeatureEngine.__new__(FeatureEngine)
        # Strong uptrend
        closes = np.linspace(2800, 2900, 30)
        rsi = engine._compute_rsi(closes, period=14)
        assert 0.0 <= rsi <= 100.0

    def test_rsi_strong_uptrend_above_50(self):
        engine = FeatureEngine.__new__(FeatureEngine)
        closes = np.linspace(2800, 2900, 30)
        rsi = engine._compute_rsi(closes, period=14)
        assert rsi > 50.0

    def test_rsi_strong_downtrend_below_50(self):
        engine = FeatureEngine.__new__(FeatureEngine)
        closes = np.linspace(2900, 2800, 30)
        rsi = engine._compute_rsi(closes, period=14)
        assert rsi < 50.0

    def test_rsi_flat_market(self):
        engine = FeatureEngine.__new__(FeatureEngine)
        closes = np.array([2850.0] * 30)
        rsi = engine._compute_rsi(closes, period=14)
        # Flat market: avg_gain=0, avg_loss=0 → avg_loss==0 → RSI=100 by convention
        # (no downward pressure, so RSI maxes out)
        assert rsi == 100.0


class TestFeatureValidation:
    def test_nan_replaced_with_zero(self):
        features = {"a": float("nan"), "b": 1.5, "c": float("inf")}
        cleaned = FeatureEngine._validate_features(features)
        assert cleaned["a"] == 0.0
        assert cleaned["b"] == 1.5
        assert cleaned["c"] == 0.0

    def test_negative_inf_replaced(self):
        features = {"x": float("-inf"), "y": 42.0}
        cleaned = FeatureEngine._validate_features(features)
        assert cleaned["x"] == 0.0
        assert cleaned["y"] == 42.0

    def test_nested_non_float_untouched(self):
        features = {"label": "LONG", "count": 5, "value": 3.14}
        cleaned = FeatureEngine._validate_features(features)
        assert cleaned["label"] == "LONG"
        assert cleaned["count"] == 5
        assert cleaned["value"] == 3.14
