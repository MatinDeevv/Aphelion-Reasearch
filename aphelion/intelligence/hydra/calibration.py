"""
APHELION HYDRA — Confidence Calibration & Disagreement (Phase 7 v2)
Isotonic regression calibration and model disagreement detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CalibrationResult:
    raw_confidence: float
    calibrated_confidence: float
    reliability_score: float  # How well-calibrated the model is [0, 1]


class IsotonicCalibrator:
    """Post-hoc confidence calibration using isotonic regression on validation set."""

    def __init__(self):
        self._fitted = False
        self._x: List[float] = []
        self._y: List[float] = []

    def fit(self, raw_confidences: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit calibrator on validation data.

        Args:
            raw_confidences: Model confidence scores [0, 1]
            outcomes: Binary outcomes (1=correct, 0=incorrect)
        """
        sorted_indices = np.argsort(raw_confidences)
        self._x = raw_confidences[sorted_indices].tolist()
        self._y = outcomes[sorted_indices].tolist()

        # Isotonic regression: pool adjacent violators
        self._y = self._pool_adjacent_violators(self._y)
        self._fitted = True

    @staticmethod
    def _pool_adjacent_violators(y: List[float]) -> List[float]:
        """Pool Adjacent Violators Algorithm for isotonic regression."""
        n = len(y)
        if n == 0:
            return y
        result = list(y)
        blocks = [[i] for i in range(n)]

        i = 0
        while i < len(blocks) - 1:
            avg_curr = np.mean([result[j] for j in blocks[i]])
            avg_next = np.mean([result[j] for j in blocks[i + 1]])
            if avg_curr > avg_next:
                # Merge blocks
                merged = blocks[i] + blocks[i + 1]
                merged_avg = np.mean([result[j] for j in merged])
                for j in merged:
                    result[j] = merged_avg
                blocks[i] = merged
                blocks.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
        return result

    def calibrate(self, raw_confidence: float) -> float:
        """Calibrate a single confidence score."""
        if not self._fitted or not self._x:
            return raw_confidence
        # Find nearest x and return corresponding y
        idx = np.searchsorted(self._x, raw_confidence)
        idx = min(idx, len(self._y) - 1)
        return float(self._y[idx])

    def calibrate_batch(self, confidences: np.ndarray) -> np.ndarray:
        """Calibrate an array of confidence scores."""
        return np.array([self.calibrate(c) for c in confidences])

    def get_reliability(self, raw_confidences: np.ndarray,
                        outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (lower = better)."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(raw_confidences)

        for i in range(n_bins):
            mask = (raw_confidences >= bins[i]) & (raw_confidences < bins[i + 1])
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            avg_conf = raw_confidences[mask].mean()
            avg_acc = outcomes[mask].mean()
            ece += (n_bin / total) * abs(avg_conf - avg_acc)

        return 1.0 - ece  # Higher = better calibrated


class DisagreementDetector:
    """Detect when ensemble models strongly disagree."""

    def __init__(self, high_threshold: float = 0.8, moderate_threshold: float = 0.6):
        self._high_threshold = high_threshold
        self._moderate_threshold = moderate_threshold

    def compute_disagreement(self, model_votes: Dict[str, int]) -> float:
        """Compute disagreement score [0, 1]. 0=unanimous, 1=complete disagreement."""
        if not model_votes:
            return 0.0
        total = len(model_votes)
        buys = sum(1 for v in model_votes.values() if v > 0)
        sells = sum(1 for v in model_votes.values() if v < 0)
        return 1.0 - abs(buys - sells) / total

    def get_weight_adjustment(self, model_votes: Dict[str, int]) -> float:
        """Return HYDRA weight multiplier based on disagreement.

        >0.8 disagreement → force FLAT (return 0.0)
        >0.6 disagreement → reduce weight 50%
        Otherwise → full weight
        """
        disagreement = self.compute_disagreement(model_votes)
        if disagreement > self._high_threshold:
            return 0.0
        elif disagreement > self._moderate_threshold:
            return 0.5
        return 1.0

    def should_force_flat(self, model_votes: Dict[str, int]) -> bool:
        """Whether disagreement is so high we should force FLAT."""
        return self.compute_disagreement(model_votes) > self._high_threshold


class DynamicEnsembleWeights:
    """Manages dynamic ensemble weights updated by SOLA/PROMETHEUS."""

    def __init__(self, model_names: List[str]):
        n = len(model_names)
        self._weights = {name: 1.0 / n for name in model_names}
        self._performance: Dict[str, List[float]] = {name: [] for name in model_names}

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def update_weights(self, performance_deltas: Dict[str, float]) -> None:
        """Update weights based on rolling performance deltas from SOLA."""
        for name, delta in performance_deltas.items():
            if name in self._weights:
                self._weights[name] = max(0.05, self._weights[name] + delta * 0.1)
        # Renormalize
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in self._weights.items()}

    def record_performance(self, model_name: str, outcome: float) -> None:
        """Record a single prediction outcome for a model."""
        if model_name in self._performance:
            self._performance[model_name].append(outcome)
            if len(self._performance[model_name]) > 500:
                self._performance[model_name] = self._performance[model_name][-500:]

    def get_rolling_accuracy(self, model_name: str, window: int = 100) -> float:
        """Compute rolling accuracy for a model."""
        history = self._performance.get(model_name, [])
        if not history:
            return 0.5
        recent = history[-window:]
        return sum(1 for x in recent if x > 0) / len(recent)

    def rebalance_from_performance(self) -> None:
        """Auto-rebalance weights based on rolling accuracy."""
        accuracies = {}
        for name in self._weights:
            accuracies[name] = self.get_rolling_accuracy(name)

        if not accuracies:
            return

        total_acc = sum(accuracies.values())
        if total_acc <= 0:
            return

        for name in self._weights:
            self._weights[name] = max(0.05, accuracies[name] / total_acc)

        total = sum(self._weights.values())
        self._weights = {k: v / total for k, v in self._weights.items()}
