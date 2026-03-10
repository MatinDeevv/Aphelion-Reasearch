"""
ATLAS — DXY Live Feed
Phase 19 — Engineering Spec v3.0

Real-time DXY feed with trend detection and gold correlation tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DXYSnapshot:
    """Point-in-time DXY reading."""
    value: float
    sma_20: float = 0.0
    trend: int = 0          # 1 strengthening, -1 weakening
    gold_bias: int = 0      # 1 bullish gold, -1 bearish gold
    correlation: float = -0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DXYLiveFeed:
    """
    Streams DXY ticks, computes rolling trend and gold correlation.
    """

    def __init__(self, sma_period: int = 20, corr_window: int = 50):
        self._sma_period = sma_period
        self._corr_window = corr_window
        self._dxy_prices: List[float] = []
        self._gold_prices: List[float] = []
        self._latest: Optional[DXYSnapshot] = None

    def on_tick(self, dxy_price: float, gold_price: float) -> DXYSnapshot:
        self._dxy_prices.append(dxy_price)
        self._gold_prices.append(gold_price)

        # Rolling SMA
        sma = (
            np.mean(self._dxy_prices[-self._sma_period:])
            if len(self._dxy_prices) >= self._sma_period
            else dxy_price
        )

        # Trend
        trend = 0
        if len(self._dxy_prices) >= 2:
            trend = 1 if self._dxy_prices[-1] > self._dxy_prices[-2] else -1

        # Correlation
        corr = -0.5
        if len(self._dxy_prices) >= self._corr_window:
            d = np.array(self._dxy_prices[-self._corr_window:])
            g = np.array(self._gold_prices[-self._corr_window:])
            if np.std(d) > 0 and np.std(g) > 0:
                corr = float(np.corrcoef(d, g)[0, 1])

        # Gold bias
        gold_bias = 0
        if trend == 1 and corr < -0.3:
            gold_bias = -1
        elif trend == -1 and corr < -0.3:
            gold_bias = 1

        snap = DXYSnapshot(
            value=dxy_price, sma_20=float(sma), trend=trend,
            gold_bias=gold_bias, correlation=corr,
        )
        self._latest = snap
        return snap

    @property
    def latest(self) -> Optional[DXYSnapshot]:
        return self._latest
