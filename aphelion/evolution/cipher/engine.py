"""
APHELION CIPHER — Feature Importance & Alpha Decay Tracker

Rolling SHAP-inspired importance tracker that:
  1. Computes permutation-based importance for every feature
  2. Tracks 30-day vs 90-day importance ratio to detect alpha decay
  3. Estimates the half-life of each feature's predictive power
  4. Schedules proactive feature retirement/replacement alerts

Spec reference:
  CIPHER-DECAY  — 30d vs 90d rolling SHAP ratio
  CIPHER-HALFLIFE — Exponential decay fitting per signal
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class CipherConfig:
    """Configuration for the CIPHER feature importance tracker."""
    # Rolling windows
    short_window: int = 30            # 30-day rolling (recent)
    long_window: int = 90             # 90-day rolling (baseline)
    update_interval_bars: int = 50    # Recompute every N bars
    # Decay detection
    decay_threshold: float = 0.50     # Ratio < 0.50 = alpha decay alert
    critical_decay_threshold: float = 0.25  # Ratio < 0.25 = urgent retirement
    # Half-life
    min_observations: int = 20        # Min data points for half-life estimation
    halflife_warning_days: float = 30.0  # Alert when remaining half-life < 30 days
    # Permutation importance
    n_permutations: int = 10          # Number of random shuffles
    min_importance: float = 0.001     # Features below this are considered dead


@dataclass
class FeatureImportance:
    """Importance metrics for a single feature."""
    name: str
    short_importance: float = 0.0    # Recent window importance
    long_importance: float = 0.0     # Baseline window importance
    decay_ratio: float = 1.0         # short / long (1.0 = stable, <1.0 = decaying)
    half_life_days: float = float("inf")  # Estimated remaining half-life
    is_decaying: bool = False
    is_critical: bool = False        # Should be retired
    is_dead: bool = False            # Below min_importance threshold
    rank: int = 0                    # Rank among all features (1 = best)
    history: list[float] = field(default_factory=list)  # Rolling importance values
    last_updated: Optional[datetime] = None

    @property
    def status(self) -> str:
        if self.is_dead:
            return "DEAD"
        if self.is_critical:
            return "CRITICAL"
        if self.is_decaying:
            return "DECAYING"
        return "HEALTHY"


@dataclass
class DecayAlert:
    """Alert generated when alpha decay is detected."""
    feature_name: str
    severity: str            # "WARNING" or "CRITICAL"
    decay_ratio: float
    half_life_days: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Permutation Importance Engine ───────────────────────────────────────────

class PermutationImportanceComputer:
    """
    Computes feature importance by measuring prediction quality degradation
    when each feature is randomly shuffled. Works with any scoring function.
    """

    def __init__(self, n_permutations: int = 10, rng: Optional[np.random.Generator] = None):
        self._n_perm = n_permutations
        self._rng = rng or np.random.default_rng(42)

    def compute(
        self,
        features: np.ndarray,      # shape (n_samples, n_features)
        targets: np.ndarray,       # shape (n_samples,)
        scorer: callable,          # (features, targets) -> float  (higher = better)
        feature_names: list[str],
    ) -> dict[str, float]:
        """
        Compute permutation importance for each feature.

        Returns dict of feature_name → importance_score (drop in performance).
        """
        if features.shape[0] < 10:
            return {name: 0.0 for name in feature_names}

        base_score = scorer(features, targets)
        importances: dict[str, float] = {}

        for col_idx, name in enumerate(feature_names):
            drops = []
            for _ in range(self._n_perm):
                shuffled = features.copy()
                self._rng.shuffle(shuffled[:, col_idx])
                perm_score = scorer(shuffled, targets)
                drops.append(base_score - perm_score)
            importances[name] = float(np.mean(drops))

        return importances


# ─── Half-Life Estimator ─────────────────────────────────────────────────────

def estimate_half_life(values: list[float], min_points: int = 10) -> float:
    """
    Estimate the half-life of a decaying signal using OLS on log-transformed
    importance values.

    Returns half-life in time-steps (same unit as input spacing).
    Returns inf if no decay detected or insufficient data.
    """
    if len(values) < min_points:
        return float("inf")

    arr = np.array(values, dtype=np.float64)
    # Filter to positive values only
    positive = arr[arr > 1e-10]
    if len(positive) < min_points:
        return float("inf")

    # OLS regression: log(importance) = a + b*t
    t = np.arange(len(positive), dtype=np.float64)
    log_vals = np.log(positive)

    # Solve y = a + b*t
    t_mean = np.mean(t)
    y_mean = np.mean(log_vals)
    numerator = np.sum((t - t_mean) * (log_vals - y_mean))
    denominator = np.sum((t - t_mean) ** 2)

    if abs(denominator) < 1e-10:
        return float("inf")

    slope = numerator / denominator

    # Half-life = -ln(2) / slope
    if slope >= 0:
        return float("inf")  # Not decaying

    half_life = -math.log(2) / slope
    return max(half_life, 0.0)


# ─── Main CIPHER Engine ─────────────────────────────────────────────────────

class CipherEngine:
    """
    CIPHER — Feature Importance & Alpha Decay Detection Engine.

    Tracks the predictive value of every feature over time,
    detects alpha decay before it impacts performance, and
    generates retirement/replacement alerts.

    Usage:
        cipher = CipherEngine(config)
        cipher.update(feature_matrix, targets, feature_names)
        alerts = cipher.get_alerts()
        rankings = cipher.get_rankings()
    """

    def __init__(self, config: Optional[CipherConfig] = None):
        self._config = config or CipherConfig()
        self._importance_computer = PermutationImportanceComputer(
            n_permutations=self._config.n_permutations
        )
        self._features: dict[str, FeatureImportance] = {}
        self._alerts: list[DecayAlert] = []
        self._bars_since_update: int = 0
        self._update_count: int = 0
        # Rolling importance history: feature_name -> list of (timestamp, importance)
        self._history_buffer: dict[str, list[tuple[datetime, float]]] = {}

    # ── Core Update ──────────────────────────────────────────────────────────

    def update(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: list[str],
        scorer: Optional[callable] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[DecayAlert]:
        """
        Update feature importance estimates.

        Args:
            features: Feature matrix (n_samples, n_features).
            targets: Target values (n_samples,).
            feature_names: Column names matching features columns.
            scorer: Scoring function (features, targets) -> float.
                    Defaults to correlation-based scorer.
            timestamp: Current timestamp. Defaults to now().

        Returns:
            List of new DecayAlerts generated.
        """
        self._bars_since_update += 1
        if self._bars_since_update < self._config.update_interval_bars and self._update_count > 0:
            return []

        self._bars_since_update = 0
        self._update_count += 1
        ts = timestamp or datetime.now(timezone.utc)

        # Default scorer: average absolute correlation with target
        if scorer is None:
            scorer = self._default_scorer

        # Compute current importances
        raw_importances = self._importance_computer.compute(
            features, targets, scorer, feature_names
        )

        # Update tracking state
        new_alerts = []
        for name, importance in raw_importances.items():
            fi = self._features.get(name)
            if fi is None:
                fi = FeatureImportance(name=name)
                self._features[name] = fi

            # Append to history
            fi.history.append(importance)
            if len(fi.history) > self._config.long_window * 2:
                fi.history = fi.history[-self._config.long_window * 2:]

            # History buffer for half-life
            if name not in self._history_buffer:
                self._history_buffer[name] = []
            self._history_buffer[name].append((ts, importance))
            if len(self._history_buffer[name]) > self._config.long_window * 3:
                self._history_buffer[name] = self._history_buffer[name][-self._config.long_window * 3:]

            fi.last_updated = ts

            # Compute short/long importance
            short_w = self._config.short_window
            long_w = self._config.long_window

            if len(fi.history) >= short_w:
                fi.short_importance = float(np.mean(fi.history[-short_w:]))
            else:
                fi.short_importance = importance

            if len(fi.history) >= long_w:
                fi.long_importance = float(np.mean(fi.history[-long_w:]))
            else:
                fi.long_importance = fi.short_importance

            # Decay ratio
            if fi.long_importance > self._config.min_importance:
                fi.decay_ratio = fi.short_importance / fi.long_importance
            else:
                fi.decay_ratio = 0.0

            # Half-life estimation
            if len(fi.history) >= self._config.min_observations:
                fi.half_life_days = estimate_half_life(
                    fi.history, self._config.min_observations
                )
            else:
                fi.half_life_days = float("inf")

            # Status classification
            fi.is_dead = fi.short_importance < self._config.min_importance
            fi.is_critical = fi.decay_ratio < self._config.critical_decay_threshold
            fi.is_decaying = fi.decay_ratio < self._config.decay_threshold

            # Generate alerts
            if fi.is_critical:
                alert = DecayAlert(
                    feature_name=name,
                    severity="CRITICAL",
                    decay_ratio=fi.decay_ratio,
                    half_life_days=fi.half_life_days,
                    message=f"Feature '{name}' critically decayed: ratio={fi.decay_ratio:.2f}, "
                            f"half-life={fi.half_life_days:.1f} steps. Schedule retirement.",
                    timestamp=ts,
                )
                new_alerts.append(alert)
            elif fi.is_decaying:
                alert = DecayAlert(
                    feature_name=name,
                    severity="WARNING",
                    decay_ratio=fi.decay_ratio,
                    half_life_days=fi.half_life_days,
                    message=f"Feature '{name}' decaying: ratio={fi.decay_ratio:.2f}, "
                            f"half-life={fi.half_life_days:.1f} steps. Monitor closely.",
                    timestamp=ts,
                )
                new_alerts.append(alert)

        # Update rankings
        sorted_features = sorted(
            self._features.values(),
            key=lambda f: f.short_importance,
            reverse=True,
        )
        for rank, fi in enumerate(sorted_features, 1):
            fi.rank = rank

        self._alerts.extend(new_alerts)
        if len(self._alerts) > 500:
            self._alerts = self._alerts[-250:]

        logger.info(
            "CIPHER update #%d | %d features | %d decaying | %d alerts",
            self._update_count, len(self._features),
            sum(1 for f in self._features.values() if f.is_decaying),
            len(new_alerts),
        )

        return new_alerts

    # ── Default Scorer ───────────────────────────────────────────────────────

    @staticmethod
    def _default_scorer(features: np.ndarray, targets: np.ndarray) -> float:
        """Default scorer: mean absolute correlation between features and targets."""
        if features.shape[0] < 5:
            return 0.0
        score = 0.0
        n_features = features.shape[1]
        for i in range(n_features):
            col = features[:, i]
            std_col = np.std(col)
            std_tgt = np.std(targets)
            if std_col > 1e-10 and std_tgt > 1e-10:
                corr = np.corrcoef(col, targets)[0, 1]
                if np.isfinite(corr):
                    score += abs(corr)
        return score / max(n_features, 1)

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_feature(self, name: str) -> Optional[FeatureImportance]:
        """Get importance metrics for a specific feature."""
        return self._features.get(name)

    def get_rankings(self, top_n: int = 0) -> list[FeatureImportance]:
        """
        Get features ranked by current importance.
        If top_n > 0, return only the top N.
        """
        ranked = sorted(
            self._features.values(),
            key=lambda f: f.short_importance,
            reverse=True,
        )
        if top_n > 0:
            return ranked[:top_n]
        return ranked

    def get_decaying_features(self) -> list[FeatureImportance]:
        """Get all features currently in decay."""
        return [f for f in self._features.values() if f.is_decaying]

    def get_dead_features(self) -> list[FeatureImportance]:
        """Get all features below minimum importance."""
        return [f for f in self._features.values() if f.is_dead]

    def get_alerts(self, severity: Optional[str] = None) -> list[DecayAlert]:
        """Get alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self._alerts if a.severity == severity]
        return list(self._alerts)

    def get_healthy_feature_names(self) -> list[str]:
        """Return names of all non-decaying, non-dead features."""
        return [f.name for f in self._features.values()
                if not f.is_decaying and not f.is_dead]

    def get_feature_weights(self) -> dict[str, float]:
        """
        Get importance-derived feature weights (normalised to sum to 1).
        Dead features get weight 0. Can be fed to MERIDIAN or HYDRA.
        """
        active = {
            name: max(fi.short_importance, 0.0)
            for name, fi in self._features.items()
            if not fi.is_dead
        }
        total = sum(active.values())
        if total <= 0:
            return {name: 1.0 / len(active) if active else 0.0 for name in active}
        return {name: v / total for name, v in active.items()}

    # ── Summary ──────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Get a summary snapshot of CIPHER state."""
        features = list(self._features.values())
        return {
            "total_features": len(features),
            "healthy": sum(1 for f in features if f.status == "HEALTHY"),
            "decaying": sum(1 for f in features if f.status == "DECAYING"),
            "critical": sum(1 for f in features if f.status == "CRITICAL"),
            "dead": sum(1 for f in features if f.status == "DEAD"),
            "total_alerts": len(self._alerts),
            "update_count": self._update_count,
            "top_5": [
                {"name": f.name, "importance": f.short_importance, "decay_ratio": f.decay_ratio}
                for f in self.get_rankings(5)
            ],
        }

    def reset(self) -> None:
        """Reset all tracked state."""
        self._features.clear()
        self._alerts.clear()
        self._history_buffer.clear()
        self._bars_since_update = 0
        self._update_count = 0
