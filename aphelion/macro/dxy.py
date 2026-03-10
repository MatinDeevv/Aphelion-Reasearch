"""
MACRO DXY Correlation Monitor
Tracks USD Index correlation with gold.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class DXYBias(Enum):
    BUY_GOLD = "BUY_GOLD"      # DXY falling → bullish gold
    SELL_GOLD = "SELL_GOLD"     # DXY rising → bearish gold
    NEUTRAL = "NEUTRAL"


@dataclass
class DXYState:
    bias: DXYBias = DXYBias.NEUTRAL
    correlation: float = -0.6     # Rolling correlation (normal: -0.6 to -0.8)
    correlation_breakdown: bool = False   # True when correlation becomes positive
    dxy_trend_1h: str = "NEUTRAL"
    dxy_trend_4h: str = "NEUTRAL"
    dxy_momentum: float = 0.0


class DXYMonitor:
    """
    DXY (USD index) typically inversely correlated with gold.
    When correlation breaks down → regime shift warning.
    """

    def __init__(self, correlation_window: int = 50, breakdown_threshold: float = -0.3):
        self._window = correlation_window
        self._breakdown = breakdown_threshold
        self._gold_returns: List[float] = []
        self._dxy_returns: List[float] = []

    def update(
        self, gold_price: float, dxy_price: float,
        prev_gold: Optional[float] = None, prev_dxy: Optional[float] = None,
    ) -> DXYState:
        """Update with latest gold and DXY prices."""
        if prev_gold and prev_gold > 0:
            self._gold_returns.append((gold_price - prev_gold) / prev_gold)
        if prev_dxy and prev_dxy > 0:
            self._dxy_returns.append((dxy_price - prev_dxy) / prev_dxy)

        # Trim to window
        self._gold_returns = self._gold_returns[-self._window * 2:]
        self._dxy_returns = self._dxy_returns[-self._window * 2:]

        correlation = self.compute_rolling_correlation()
        breakdown = correlation > self._breakdown

        # Determine DXY trend from returns
        if len(self._dxy_returns) >= 5:
            recent_dxy = sum(self._dxy_returns[-5:])
            if recent_dxy > 0.001:
                trend = "UP"
            elif recent_dxy < -0.001:
                trend = "DOWN"
            else:
                trend = "NEUTRAL"
        else:
            trend = "NEUTRAL"

        # DXY momentum
        momentum = sum(self._dxy_returns[-10:]) if len(self._dxy_returns) >= 10 else 0.0

        # Bias
        if trend == "DOWN" and not breakdown:
            bias = DXYBias.BUY_GOLD
        elif trend == "UP" and not breakdown:
            bias = DXYBias.SELL_GOLD
        else:
            bias = DXYBias.NEUTRAL

        return DXYState(
            bias=bias,
            correlation=correlation,
            correlation_breakdown=breakdown,
            dxy_trend_1h=trend,
            dxy_trend_4h=trend,  # Simplified; would use different timeframe data
            dxy_momentum=momentum,
        )

    def compute_rolling_correlation(self) -> float:
        """Compute rolling correlation between gold and DXY returns."""
        n = min(len(self._gold_returns), len(self._dxy_returns), self._window)
        if n < 5:
            return -0.6  # Default assumption

        gold = np.array(self._gold_returns[-n:])
        dxy = np.array(self._dxy_returns[-n:])

        if np.std(gold) == 0 or np.std(dxy) == 0:
            return 0.0

        corr = float(np.corrcoef(gold, dxy)[0, 1])
        return corr if not np.isnan(corr) else 0.0

    def detect_correlation_breakdown(self) -> bool:
        correlation = self.compute_rolling_correlation()
        return correlation > self._breakdown

    def reset(self) -> None:
        self._gold_returns.clear()
        self._dxy_returns.clear()
