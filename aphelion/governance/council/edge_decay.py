"""
SOLA — Edge Decay Tracker (Standalone)
Phase 21 — Engineering Spec v3.0

Dedicated edge-decay monitor with CUSUM + rolling metrics.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class EdgeDecayTracker:
    """
    CUSUM-based edge decay detection.
    Detects when trading edge is eroding via cumulative sum of
    deviations from expected return mean.
    """

    def __init__(self, cusum_threshold: float = 2.0, min_trades: int = 50):
        self._threshold = cusum_threshold
        self._min_trades = min_trades
        self._returns: List[float] = []
        self._cusum: float = 0.0
        self._target_mean: float = 0.0
        self._calibrated: bool = False
        self._decay_active: bool = False

    def calibrate(self, returns: List[float]) -> None:
        if len(returns) >= self._min_trades:
            self._target_mean = sum(returns) / len(returns)
            self._calibrated = True

    def update(self, trade_return: float) -> bool:
        self._returns.append(trade_return)
        if not self._calibrated and len(self._returns) >= self._min_trades:
            self.calibrate(self._returns[:self._min_trades])
        if not self._calibrated:
            return False

        deviation = self._target_mean - trade_return
        self._cusum = max(0, self._cusum + deviation)
        self._decay_active = self._cusum > self._threshold
        return self._decay_active

    @property
    def decay_active(self) -> bool:
        return self._decay_active

    @property
    def cusum_value(self) -> float:
        return self._cusum

    @property
    def trade_count(self) -> int:
        return len(self._returns)

    def reset(self) -> None:
        self._cusum = 0.0
        self._decay_active = False
