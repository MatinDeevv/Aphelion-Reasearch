"""
APHELION Cointegration Features
Engle-Granger cointegration tests, rolling spread z-scores.
Used by VENOM statistical arbitrage engine.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from statsmodels.tsa.stattools import adfuller


@dataclass
class CointegrationResult:
    pair: str
    is_cointegrated: bool
    p_value: float
    spread_zscore: float
    half_life: float
    hedge_ratio: float


class CointegrationEngine:
    """Rolling cointegration analysis for gold vs correlated assets."""

    PAIRS = [
        ("XAUUSD", "DXY"),
        ("XAUUSD", "REAL_YIELD"),
        ("XAUUSD", "XAGUSD"),  # Gold/Silver ratio
    ]

    def __init__(self, window: int = 50, p_value_threshold: float = 0.05):
        self._window = window
        self._p_threshold = p_value_threshold
        self._results: dict[str, CointegrationResult] = {}

    def _ols_residuals(self, y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Simple OLS regression. Returns residuals and hedge ratio (beta)."""
        x_with_const = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.zeros(len(y)), 1.0
        residuals = y - x_with_const @ beta
        return residuals, beta[1]

    def _adf_test_simple(self, series: np.ndarray) -> float:
        """
        ADF test using statsmodels. Returns p-value.
        """
        if len(series) < 10:
            return 1.0

        if np.std(series) == 0:
            return 1.0

        try:
            result = adfuller(series)
            return result[1]  # p-value
        except Exception:
            return 1.0

    def _half_life(self, spread: np.ndarray) -> float:
        """Estimate mean-reversion half-life via OLS on lagged spread."""
        if len(spread) < 10:
            return float('inf')

        lagged = spread[:-1]
        diff = np.diff(spread)

        if np.std(lagged) == 0:
            return float('inf')

        x = np.column_stack([np.ones(len(lagged)), lagged])
        try:
            beta = np.linalg.lstsq(x, diff, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float('inf')

        lam = beta[1]
        if lam >= 0:
            return float('inf')

        return -np.log(2) / lam

    def test_pair(self, y: np.ndarray, x: np.ndarray,
                  pair_name: str) -> CointegrationResult:
        """Run Engle-Granger cointegration test on a pair."""
        residuals, hedge_ratio = self._ols_residuals(y, x)
        p_value = self._adf_test_simple(residuals)
        is_coint = p_value < self._p_threshold

        # Spread z-score
        spread_mean = np.mean(residuals)
        spread_std = np.std(residuals)
        if spread_std > 0:
            zscore = (residuals[-1] - spread_mean) / spread_std
        else:
            zscore = 0.0

        half_life = self._half_life(residuals)

        result = CointegrationResult(
            pair=pair_name,
            is_cointegrated=is_coint,
            p_value=p_value,
            spread_zscore=zscore,
            half_life=half_life,
            hedge_ratio=hedge_ratio,
        )
        self._results[pair_name] = result
        return result

    def compute_all(self, data: dict[str, np.ndarray]) -> dict:
        """
        Compute cointegration for all configured pairs.
        data: {"XAUUSD": prices, "DXY": prices, "REAL_YIELD": prices, "XAGUSD": prices}
        """
        results = {}

        for y_sym, x_sym in self.PAIRS:
            if y_sym in data and x_sym in data:
                y = data[y_sym][-self._window:]
                x = data[x_sym][-self._window:]
                min_len = min(len(y), len(x))
                if min_len >= 20:
                    pair_name = f"{y_sym}_vs_{x_sym}"
                    result = self.test_pair(y[-min_len:], x[-min_len:], pair_name)
                    results[pair_name] = {
                        "cointegrated": result.is_cointegrated,
                        "p_value": result.p_value,
                        "spread_zscore": result.spread_zscore,
                        "half_life": result.half_life,
                        "hedge_ratio": result.hedge_ratio,
                    }

        # Summary features
        any_coint = any(r["cointegrated"] for r in results.values()) if results else False
        max_zscore = max((abs(r["spread_zscore"]) for r in results.values()), default=0)

        return {
            "cointegration_pairs": results,
            "any_cointegrated": any_coint,
            "max_spread_zscore": max_zscore,
        }
