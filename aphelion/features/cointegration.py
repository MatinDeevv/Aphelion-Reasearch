"""
APHELION Cointegration Features  (v3 — upgraded)
Engle-Granger cointegration tests, Johansen multivariate test,
rolling spread z-scores, half-life estimation, Hurst-based mean-reversion score.

Algorithms:
  - Engle-Granger two-step (ADF on OLS residuals)
  - Johansen trace/max-eigenvalue test (statsmodels)
  - Ornstein-Uhlenbeck half-life via OLS on lagged spread
  - Hurst exponent on spread for mean-reversion confirmation
  - Rolling z-score with exponential weighting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from statsmodels.tsa.stattools import adfuller

try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    _HAS_JOHANSEN = True
except ImportError:
    _HAS_JOHANSEN = False


@dataclass
class CointegrationResult:
    pair: str
    is_cointegrated: bool
    p_value: float
    spread_zscore: float
    half_life: float
    hedge_ratio: float
    hurst_exponent: float = 0.5          # H < 0.5 → mean-reverting
    ewm_zscore: float = 0.0             # Exponentially weighted z-score
    johansen_trace_stat: float = 0.0    # Johansen trace statistic (if available)
    johansen_coint: bool = False        # Johansen test result


class CointegrationEngine:
    """Rolling cointegration analysis for gold vs correlated assets (v3)."""

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
        """ADF test using statsmodels. Returns p-value."""
        if len(series) < 10:
            return 1.0
        if np.std(series) == 0:
            return 1.0
        try:
            result = adfuller(series)
            return result[1]
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

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """
        Rescaled Range (R/S) Hurst exponent on spread.
        H < 0.5 → mean-reverting (good for stat-arb)
        H ≈ 0.5 → random walk
        H > 0.5 → trending
        """
        n = len(series)
        if n < 20:
            return 0.5

        max_k = min(n // 2, 100)
        sizes = []
        rs_values = []

        for size in range(10, max_k + 1, max(1, (max_k - 10) // 10)):
            n_chunks = n // size
            if n_chunks < 1:
                continue

            rs_list = []
            for i in range(n_chunks):
                chunk = series[i * size:(i + 1) * size]
                mean_c = np.mean(chunk)
                devs = np.cumsum(chunk - mean_c)
                r = np.max(devs) - np.min(devs)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                sizes.append(size)
                rs_values.append(np.mean(rs_list))

        if len(sizes) < 2:
            return 0.5

        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)
        try:
            slope = np.polyfit(log_sizes, log_rs, 1)[0]
            return float(np.clip(slope, 0.0, 1.0))
        except Exception:
            return 0.5

    def _ewm_zscore(self, residuals: np.ndarray, span: int = 20) -> float:
        """Exponentially weighted z-score — recency-biased spread signal."""
        if len(residuals) < 5:
            return 0.0
        series = pd.Series(residuals)
        ewm_mean = series.ewm(span=span, min_periods=5).mean().iloc[-1]
        ewm_std = series.ewm(span=span, min_periods=5).std().iloc[-1]
        if ewm_std == 0 or np.isnan(ewm_std):
            return 0.0
        return float((residuals[-1] - ewm_mean) / ewm_std)

    def _johansen_test(self, y: np.ndarray, x: np.ndarray) -> tuple[float, bool]:
        """
        Johansen cointegration test (trace statistic).
        Returns (trace_stat, is_cointegrated).
        """
        if not _HAS_JOHANSEN or len(y) < 20:
            return 0.0, False

        try:
            data = np.column_stack([y, x])
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            # Compare trace stat for r=0 against 95% critical value
            trace_stat = float(result.lr1[0])
            crit_95 = float(result.cvt[0, 1])  # 95% critical value for r=0
            return trace_stat, trace_stat > crit_95
        except Exception:
            return 0.0, False

    def test_pair(self, y: np.ndarray, x: np.ndarray,
                  pair_name: str) -> CointegrationResult:
        """Run Engle-Granger + Johansen cointegration tests on a pair."""
        residuals, hedge_ratio = self._ols_residuals(y, x)
        p_value = self._adf_test_simple(residuals)
        is_coint = p_value < self._p_threshold

        # Spread z-score (simple)
        spread_mean = np.mean(residuals)
        spread_std = np.std(residuals)
        if spread_std > 0:
            zscore = (residuals[-1] - spread_mean) / spread_std
        else:
            zscore = 0.0

        half_life = self._half_life(residuals)

        # v3: Hurst exponent on residuals
        hurst = self._hurst_exponent(residuals)

        # v3: EWM z-score
        ewm_z = self._ewm_zscore(residuals)

        # v3: Johansen test
        johansen_stat, johansen_coint = self._johansen_test(y, x)

        # Combined: cointegrated if EITHER Engle-Granger OR Johansen says so
        combined_coint = is_coint or johansen_coint

        result = CointegrationResult(
            pair=pair_name,
            is_cointegrated=combined_coint,
            p_value=p_value,
            spread_zscore=zscore,
            half_life=half_life,
            hedge_ratio=hedge_ratio,
            hurst_exponent=hurst,
            ewm_zscore=ewm_z,
            johansen_trace_stat=johansen_stat,
            johansen_coint=johansen_coint,
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
                        "hurst_exponent": result.hurst_exponent,
                        "ewm_zscore": result.ewm_zscore,
                        "johansen_trace_stat": result.johansen_trace_stat,
                        "johansen_coint": result.johansen_coint,
                        "mean_reverting": result.hurst_exponent < 0.45,
                    }

        # Summary features
        any_coint = any(r["cointegrated"] for r in results.values()) if results else False
        max_zscore = max((abs(r["spread_zscore"]) for r in results.values()), default=0)
        # v3: best mean-reversion candidate
        best_mr = min(
            (r["hurst_exponent"] for r in results.values()),
            default=0.5,
        )

        return {
            "cointegration_pairs": results,
            "any_cointegrated": any_coint,
            "max_spread_zscore": max_zscore,
            "best_hurst": best_mr,
        }
