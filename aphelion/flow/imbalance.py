"""
FLOW Bid-Ask Imbalance Tracker
Detects aggressive buying/selling pressure.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np


@dataclass
class ImbalanceState:
    imbalance: float = 0.0       # [-1, 1] — positive = buying pressure
    rolling_imbalance: float = 0.0
    imbalance_z_score: float = 0.0
    extreme: bool = False        # True if z-score > 2.0


class ImbalanceTracker:
    """
    Track bid-ask volume imbalance to detect institutional activity.
    High imbalance = aggressive directional pressure.
    """

    def __init__(self, window: int = 50, z_threshold: float = 2.0):
        self._window = window
        self._z_threshold = z_threshold
        self._history: deque = deque(maxlen=window)
        self._state = ImbalanceState()

    def update(self, buy_volume: float, sell_volume: float) -> ImbalanceState:
        """Update with new volume split."""
        total = buy_volume + sell_volume
        if total <= 0:
            return self._state

        imbalance = (buy_volume - sell_volume) / total
        self._history.append(imbalance)

        if len(self._history) >= 5:
            arr = np.array(self._history)
            rolling = float(np.mean(arr))
            std = float(np.std(arr))
            z_score = (imbalance - rolling) / std if std > 0 else 0.0
        else:
            rolling = imbalance
            z_score = 0.0

        self._state = ImbalanceState(
            imbalance=imbalance,
            rolling_imbalance=rolling,
            imbalance_z_score=z_score,
            extreme=abs(z_score) > self._z_threshold,
        )
        return self._state

    @property
    def state(self) -> ImbalanceState:
        return self._state

    def reset(self) -> None:
        self._history.clear()
        self._state = ImbalanceState()
