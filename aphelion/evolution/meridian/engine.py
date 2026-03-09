"""
APHELION MERIDIAN — Dynamic Multi-Timeframe Weighting

Sub-components:
  MERIDIAN-GRANGER  – Rolling Granger causality F-statistics between timeframe pairs
  MERIDIAN-WEIGHTS  – Dynamic weight vector [w_1m, w_5m, w_15m, w_1h]

Replaces static MTF alignment with causality-derived weights.
Updated every 50 bars. Feeds directly into MTFAlignmentEngine.set_weights().

Spec reference:
  "MERIDIAN-GRANGER: Rolling Granger causality F-statistics between all
   timeframe pairs. Determines which timeframe is currently driving price."
  "MERIDIAN-WEIGHTS: Dynamic timeframe weight vector … updated every 50 bars."
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aphelion.core.config import Timeframe, TIMEFRAMES

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class MeridianConfig:
    """Configuration for the MERIDIAN MTF weighting engine."""
    # Granger
    granger_window: int = 200          # Rolling window for Granger test
    granger_max_lag: int = 5           # Max lag for causality test
    # Weight update
    update_interval_bars: int = 50     # Recompute weights every N bars
    min_weight: float = 0.05           # Floor — no TF gets less than 5%
    max_weight: float = 0.60           # Ceiling — no TF gets more than 60%
    smoothing_alpha: float = 0.3       # EMA smoothing for weight transitions
    # Quality
    min_samples: int = 50              # Min bars before Granger is meaningful
    significance_level: float = 0.05   # p-value threshold for significance


@dataclass
class GrangerResult:
    """Result of a Granger causality test between two timeframes."""
    cause_tf: Timeframe
    effect_tf: Timeframe
    f_statistic: float
    p_value: float
    lag: int
    is_significant: bool


@dataclass
class MeridianState:
    """Current state of the MERIDIAN engine."""
    weights: dict[str, float] = field(default_factory=dict)   # tf.value -> weight
    causality_matrix: dict[str, dict[str, float]] = field(default_factory=dict)  # cause -> effect -> F
    dominant_timeframe: str = ""
    update_count: int = 0
    last_granger_results: list[dict] = field(default_factory=list)


# ─── Granger Causality Engine ────────────────────────────────────────────────

def _ols_residual_ss(y: np.ndarray, X: np.ndarray) -> float:
    """Compute residual sum of squares from OLS regression y ~ X."""
    if X.shape[0] <= X.shape[1]:
        return float("inf")
    # Normal equations: beta = (X'X)^-1 X'y
    try:
        XtX = X.T @ X
        # Add small regularisation for numerical stability
        XtX += np.eye(XtX.shape[0]) * 1e-8
        beta = np.linalg.solve(XtX, X.T @ y)
        residuals = y - X @ beta
        return float(np.sum(residuals ** 2))
    except np.linalg.LinAlgError:
        return float("inf")


def granger_causality_f(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 5,
) -> tuple[float, float, int]:
    """
    Test whether x Granger-causes y using an F-test.

    Restricted model:  y_t = a0 + sum(a_i * y_{t-i})
    Unrestricted model: y_t = a0 + sum(a_i * y_{t-i}) + sum(b_j * x_{t-j})

    Returns (best_F_statistic, best_p_value, best_lag).
    Tests lags 1..max_lag and returns the lag with highest F.
    """
    n = min(len(x), len(y))
    if n < max_lag + 10:
        return 0.0, 1.0, 1

    best_f, best_p, best_lag = 0.0, 1.0, 1

    for lag in range(1, max_lag + 1):
        # Build regression matrices
        y_target = y[lag:]
        n_obs = len(y_target)
        if n_obs <= 2 * lag + 1:
            continue

        # Restricted: y_t ~ const + y_{t-1} ... y_{t-lag}
        X_restricted = np.ones((n_obs, lag + 1))
        for i in range(1, lag + 1):
            X_restricted[:, i] = y[lag - i: lag - i + n_obs]

        rss_r = _ols_residual_ss(y_target, X_restricted)

        # Unrestricted: y_t ~ const + y_{t-1}..y_{t-lag} + x_{t-1}..x_{t-lag}
        X_unrestricted = np.ones((n_obs, 2 * lag + 1))
        for i in range(1, lag + 1):
            X_unrestricted[:, i] = y[lag - i: lag - i + n_obs]
            X_unrestricted[:, lag + i] = x[lag - i: lag - i + n_obs]

        rss_u = _ols_residual_ss(y_target, X_unrestricted)

        # F-statistic
        df1 = lag  # extra parameters
        df2 = n_obs - 2 * lag - 1
        if df2 <= 0 or rss_u <= 0:
            continue

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        f_stat = max(f_stat, 0.0)

        # Approximate p-value using F-distribution survival function
        # Use the incomplete beta function approximation
        p_value = _f_survival(f_stat, df1, df2)

        if f_stat > best_f:
            best_f = f_stat
            best_p = p_value
            best_lag = lag

    return best_f, best_p, best_lag


def _f_survival(f_stat: float, df1: int, df2: int) -> float:
    """
    Approximate survival function of F-distribution.
    Uses the regularised incomplete beta function relation:
      P(F > f) = I_{x}(df2/2, df1/2)  where x = df2/(df2 + df1*f)
    Approximated via continued fraction for numerical stability.
    """
    if f_stat <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f_stat)
    a = df2 / 2.0
    b = df1 / 2.0
    # Regularised incomplete beta I_x(a, b) ≈ crude approximation
    # For a proper implementation we'd use scipy, but we keep it dependency-free
    return _regularized_incomplete_beta(x, a, b)


def _regularized_incomplete_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """
    Compute regularised incomplete beta function I_x(a, b) using
    Lentz's continued fraction algorithm.
    """
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use the continued fraction representation that converges better
    # when x < (a+1)/(a+b+2), otherwise use the symmetry relation
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a, max_iter)

    # Log beta function for normalisation
    try:
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - log_beta) / a
    except (ValueError, OverflowError):
        return 0.5

    # Lentz's continued fraction
    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(max_iter):
        if m == 0:
            numerator = 1.0
        elif m % 2 == 1:
            k = (m - 1) // 2 + 1
            numerator = -(a + k - 1 + k) * (a + k) * x / ((a + 2 * k - 1) * (a + 2 * k))
            # Correction for b
            numerator = k * (b - k) * x / ((a + 2 * k - 1) * (a + 2 * k))
        else:
            k = m // 2
            numerator = k * (b - k) * x / ((a + 2 * k - 1) * (a + 2 * k))

        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny

        f *= c * d

        if abs(c * d - 1.0) < 1e-10:
            break

    return front * (f - 1.0) if front * (f - 1.0) >= 0 else 0.5


# ─── Main MERIDIAN Engine ───────────────────────────────────────────────────

class MeridianEngine:
    """
    MERIDIAN — Dynamic Multi-Timeframe Weighting Engine.

    Computes rolling Granger causality between all timeframe pairs
    to determine which TF is currently driving price, then produces
    a dynamic weight vector that replaces static MTF alignment.

    Usage:
        meridian = MeridianEngine(config)
        weights = meridian.update(bars_by_tf)
        feature_engine.set_mtf_weights(weights)  # Feed to MTFAlignmentEngine
    """

    def __init__(self, config: Optional[MeridianConfig] = None):
        self._config = config or MeridianConfig()
        self._timeframes = list(TIMEFRAMES)  # [M1, M5, M15, H1]

        # Current weights (start with equal weighting)
        n = len(self._timeframes)
        self._weights: dict[Timeframe, float] = {
            tf: 1.0 / n for tf in self._timeframes
        }

        # Granger causality results
        self._causality_matrix: dict[tuple[Timeframe, Timeframe], GrangerResult] = {}
        self._bars_since_update: int = 0
        self._update_count: int = 0

        # Price history per timeframe (rolling buffer)
        self._price_buffers: dict[Timeframe, list[float]] = {
            tf: [] for tf in self._timeframes
        }

    # ── Feed Data ────────────────────────────────────────────────────────────

    def feed_bar(self, timeframe: Timeframe, close_price: float) -> None:
        """Feed a new close price for a specific timeframe."""
        buf = self._price_buffers.get(timeframe)
        if buf is not None:
            buf.append(close_price)
            max_len = self._config.granger_window * 2
            if len(buf) > max_len:
                self._price_buffers[timeframe] = buf[-max_len:]

    # ── Core Update ──────────────────────────────────────────────────────────

    def update(
        self,
        bars_by_tf: Optional[dict[Timeframe, np.ndarray]] = None,
    ) -> dict[Timeframe, float]:
        """
        Update Granger causality analysis and recompute weights.

        Args:
            bars_by_tf: Optional dict of Timeframe -> close prices array.
                        If provided, replaces internal buffers.

        Returns:
            Current weight dictionary.
        """
        self._bars_since_update += 1

        # Populate from explicit data if given
        if bars_by_tf is not None:
            for tf, prices in bars_by_tf.items():
                if tf in self._price_buffers:
                    self._price_buffers[tf] = list(prices)

        # Only recompute at the specified interval
        if self._bars_since_update < self._config.update_interval_bars and self._update_count > 0:
            return dict(self._weights)

        self._bars_since_update = 0
        self._update_count += 1

        # Build returns arrays for each timeframe
        returns_by_tf: dict[Timeframe, np.ndarray] = {}
        for tf in self._timeframes:
            prices = self._price_buffers.get(tf, [])
            if len(prices) >= self._config.min_samples:
                arr = np.array(prices[-self._config.granger_window:], dtype=np.float64)
                # Log returns
                with np.errstate(divide="ignore", invalid="ignore"):
                    rets = np.diff(np.log(arr))
                    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
                returns_by_tf[tf] = rets

        if len(returns_by_tf) < 2:
            return dict(self._weights)

        # Compute pairwise Granger causality
        causality_strengths: dict[Timeframe, float] = {tf: 0.0 for tf in self._timeframes}

        for cause_tf in self._timeframes:
            if cause_tf not in returns_by_tf:
                continue
            for effect_tf in self._timeframes:
                if effect_tf == cause_tf or effect_tf not in returns_by_tf:
                    continue

                # Align lengths
                x = returns_by_tf[cause_tf]
                y = returns_by_tf[effect_tf]
                min_len = min(len(x), len(y))
                if min_len < self._config.min_samples:
                    continue
                x = x[-min_len:]
                y = y[-min_len:]

                f_stat, p_value, lag = granger_causality_f(
                    x, y, self._config.granger_max_lag
                )

                result = GrangerResult(
                    cause_tf=cause_tf,
                    effect_tf=effect_tf,
                    f_statistic=f_stat,
                    p_value=p_value,
                    lag=lag,
                    is_significant=p_value < self._config.significance_level,
                )
                self._causality_matrix[(cause_tf, effect_tf)] = result

                # Accumulate causality strength for the cause TF
                if result.is_significant:
                    causality_strengths[cause_tf] += f_stat

        # Convert strengths to weights
        self._compute_weights(causality_strengths)

        logger.info(
            "MERIDIAN update #%d | weights: %s | dominant: %s",
            self._update_count,
            {tf.value: f"{w:.3f}" for tf, w in self._weights.items()},
            self.dominant_timeframe.value if self.dominant_timeframe else "NONE",
        )

        return dict(self._weights)

    # ── Weight Computation ───────────────────────────────────────────────────

    def _compute_weights(self, strengths: dict[Timeframe, float]) -> None:
        """Convert Granger causality strengths into a normalised weight vector."""
        total = sum(strengths.values())

        if total <= 0:
            # No significant causality — revert to equal weights
            n = len(self._timeframes)
            new_weights = {tf: 1.0 / n for tf in self._timeframes}
        else:
            new_weights = {tf: strengths.get(tf, 0.0) / total for tf in self._timeframes}

        # Enforce floor / ceiling
        for tf in self._timeframes:
            new_weights[tf] = max(self._config.min_weight, min(self._config.max_weight, new_weights[tf]))

        # Re-normalise
        wt = sum(new_weights.values())
        if wt > 0:
            new_weights = {tf: w / wt for tf, w in new_weights.items()}

        # EMA smoothing to avoid jarring weight transitions
        alpha = self._config.smoothing_alpha
        for tf in self._timeframes:
            old_w = self._weights.get(tf, new_weights[tf])
            self._weights[tf] = alpha * new_weights[tf] + (1 - alpha) * old_w

        # Final normalisation
        wt = sum(self._weights.values())
        if wt > 0:
            self._weights = {tf: w / wt for tf, w in self._weights.items()}

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def weights(self) -> dict[Timeframe, float]:
        """Current timeframe weight vector."""
        return dict(self._weights)

    @property
    def dominant_timeframe(self) -> Optional[Timeframe]:
        """The timeframe with the highest current weight."""
        if not self._weights:
            return None
        return max(self._weights, key=self._weights.get)  # type: ignore

    def get_causality_matrix(self) -> dict[str, dict[str, float]]:
        """
        Get the causality matrix as a nested dict:
          { cause_tf.value: { effect_tf.value: F_statistic } }
        """
        matrix: dict[str, dict[str, float]] = {}
        for (cause, effect), result in self._causality_matrix.items():
            if cause.value not in matrix:
                matrix[cause.value] = {}
            matrix[cause.value][effect.value] = result.f_statistic
        return matrix

    def get_granger_results(self) -> list[GrangerResult]:
        """Get all Granger causality test results."""
        return list(self._causality_matrix.values())

    def get_significant_causalities(self) -> list[GrangerResult]:
        """Get only statistically significant causality relationships."""
        return [r for r in self._causality_matrix.values() if r.is_significant]

    def get_state(self) -> MeridianState:
        """Get full engine state snapshot."""
        return MeridianState(
            weights={tf.value: w for tf, w in self._weights.items()},
            causality_matrix=self.get_causality_matrix(),
            dominant_timeframe=self.dominant_timeframe.value if self.dominant_timeframe else "",
            update_count=self._update_count,
            last_granger_results=[
                {
                    "cause": r.cause_tf.value,
                    "effect": r.effect_tf.value,
                    "f_stat": r.f_statistic,
                    "p_value": r.p_value,
                    "lag": r.lag,
                    "significant": r.is_significant,
                }
                for r in self._causality_matrix.values()
            ],
        )

    def reset(self) -> None:
        """Reset all state."""
        n = len(self._timeframes)
        self._weights = {tf: 1.0 / n for tf in self._timeframes}
        self._causality_matrix.clear()
        self._price_buffers = {tf: [] for tf in self._timeframes}
        self._bars_since_update = 0
        self._update_count = 0
