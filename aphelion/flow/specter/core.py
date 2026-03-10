"""
SPECTER — Dark Pool / Stealth Flow Detector Core
Phase 18 — Engineering Spec v3.0

Identifies institutional stealth accumulation/distribution patterns
from order flow imbalances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StealthSignal:
    """Detected stealth accumulation/distribution."""
    direction: int          # 1 = accumulation, -1 = distribution
    intensity: float        # [0, 1]
    volume_imbalance: float # Buy/Sell volume ratio deviation
    confidence: float       # [0, 1]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SpecterCore:
    """
    Detects stealth institutional flow via:
    1. Volume imbalance that diverges from price movement
    2. Consistent one-sided absorption at key levels
    3. Time-weighted order flow asymmetry
    """

    def __init__(self, lookback: int = 50, imbalance_threshold: float = 0.3):
        self._lookback = lookback
        self._imbalance_thresh = imbalance_threshold
        self._buy_volumes: List[float] = []
        self._sell_volumes: List[float] = []
        self._price_changes: List[float] = []
        self._signals: List[StealthSignal] = []

    def update(
        self,
        buy_volume: float,
        sell_volume: float,
        price_change: float,
    ) -> Optional[StealthSignal]:
        """Update with bar-level buy/sell volume and price change."""
        self._buy_volumes.append(buy_volume)
        self._sell_volumes.append(sell_volume)
        self._price_changes.append(price_change)

        if len(self._buy_volumes) < self._lookback:
            return None

        # Recent window
        buys = np.array(self._buy_volumes[-self._lookback:])
        sells = np.array(self._sell_volumes[-self._lookback:])
        prices = np.array(self._price_changes[-self._lookback:])

        total = buys + sells
        total = np.where(total == 0, 1, total)
        imbalance = (buys - sells) / total

        # Detect: volume imbalance contradicts price → stealth flow
        avg_imbalance = float(np.mean(imbalance[-10:]))
        avg_price = float(np.mean(prices[-10:]))

        # Accumulation: buying pressure + flat/down price
        if avg_imbalance > self._imbalance_thresh and avg_price <= 0:
            signal = StealthSignal(
                direction=1,
                intensity=min(1.0, abs(avg_imbalance)),
                volume_imbalance=avg_imbalance,
                confidence=min(1.0, abs(avg_imbalance) / self._imbalance_thresh * 0.5),
            )
            self._signals.append(signal)
            return signal

        # Distribution: selling pressure + flat/up price
        if avg_imbalance < -self._imbalance_thresh and avg_price >= 0:
            signal = StealthSignal(
                direction=-1,
                intensity=min(1.0, abs(avg_imbalance)),
                volume_imbalance=avg_imbalance,
                confidence=min(1.0, abs(avg_imbalance) / self._imbalance_thresh * 0.5),
            )
            self._signals.append(signal)
            return signal

        return None

    @property
    def recent_signals(self) -> List[StealthSignal]:
        return self._signals[-20:]
