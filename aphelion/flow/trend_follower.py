"""
OMEGA — Trend Follower
Phase 17 — Engineering Spec v3.0

Trend identification and bias determination for OMEGA swing strategy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrendState:
    """Current trend analysis state."""
    direction: str    # "BULL", "BEAR", "FLAT"
    strength: float   # [0, 1]
    ema_fast: float
    ema_slow: float
    adx: float
    trend_duration_bars: int = 0


class TrendFollower:
    """
    Multi-timeframe trend identification for OMEGA.
    Uses EMA crossover confirmed by ADX strength.
    """

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        min_adx: float = 25.0,
    ):
        self._fast = fast_period
        self._slow = slow_period
        self._min_adx = min_adx
        self._trend_duration: int = 0
        self._last_direction: str = "FLAT"

    def analyze(self, closes: np.ndarray, adx: float) -> TrendState:
        """Analyze current trend state."""
        if len(closes) < self._slow:
            return TrendState("FLAT", 0.0, 0.0, 0.0, adx)

        ema_fast = self._ema(closes, self._fast)
        ema_slow = self._ema(closes, self._slow)

        if ema_fast > ema_slow and adx >= self._min_adx:
            direction = "BULL"
            strength = min(1.0, (ema_fast - ema_slow) / ema_slow * 100)
        elif ema_fast < ema_slow and adx >= self._min_adx:
            direction = "BEAR"
            strength = min(1.0, (ema_slow - ema_fast) / ema_slow * 100)
        else:
            direction = "FLAT"
            strength = 0.0

        if direction == self._last_direction and direction != "FLAT":
            self._trend_duration += 1
        else:
            self._trend_duration = 1 if direction != "FLAT" else 0
        self._last_direction = direction

        return TrendState(
            direction=direction,
            strength=strength,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            adx=adx,
            trend_duration_bars=self._trend_duration,
        )

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        mult = 2.0 / (period + 1)
        ema = float(data[0])
        for p in data[1:]:
            ema = (float(p) - ema) * mult + ema
        return ema
