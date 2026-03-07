"""Tests for APHELION Cointegration Features."""

import numpy as np
from aphelion.features.cointegration import CointegrationEngine


class TestCointegration:
    def test_cointegrated_pair(self):
        engine = CointegrationEngine(window=100)
        np.random.seed(42)

        # Create cointegrated pair: y = 2*x + noise
        x = np.cumsum(np.random.randn(100))
        y = 2 * x + np.random.randn(100) * 0.5

        result = engine.test_pair(y, x, "test_pair")
        # Should detect cointegration (strong linear relationship)
        assert result.hedge_ratio > 0

    def test_non_cointegrated(self):
        engine = CointegrationEngine(window=100)
        np.random.seed(42)

        # Two independent random walks
        x = np.cumsum(np.random.randn(100))
        y = np.cumsum(np.random.randn(100))

        result = engine.test_pair(y, x, "random_pair")
        # p-value should be high (not cointegrated)
        assert result.p_value >= 0.05

    def test_spread_zscore(self):
        engine = CointegrationEngine()
        np.random.seed(42)
        x = np.cumsum(np.random.randn(50))
        y = 1.5 * x + np.random.randn(50) * 0.3

        result = engine.test_pair(y, x, "test")
        # Z-score should be finite
        assert np.isfinite(result.spread_zscore)

    def test_compute_all(self):
        engine = CointegrationEngine(window=50)
        np.random.seed(42)
        data = {
            "XAUUSD": np.cumsum(np.random.randn(60)) + 2850,
            "DXY": np.cumsum(np.random.randn(60)) + 104,
            "REAL_YIELD": np.random.randn(60) * 0.5 + 2.0,
            "XAGUSD": np.cumsum(np.random.randn(60)) + 32,
        }
        result = engine.compute_all(data)
        assert "any_cointegrated" in result
        assert "max_spread_zscore" in result
