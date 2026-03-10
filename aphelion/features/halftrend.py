"""
APHELION HalfTrend Indicator
Trend-following indicator with low noise. Commander-tier ARES voter.
Phase 1 v2 — Engineering Spec v3.0
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HalfTrendState:
    signal: int = 0            # 1=UP, -1=DOWN, 0=FLAT
    upper_band: float = 0.0
    lower_band: float = 0.0
    strength: float = 0.0
    trend: str = "FLAT"
    atr: float = 0.0


class HalfTrendCalculator:
    """
    HalfTrend: trend-following indicator.
    Amplitude determines sensitivity, ATR period controls band width.
    """

    def __init__(self, amplitude: int = 2, atr_period: int = 100):
        self._amplitude = amplitude
        self._atr_period = atr_period

    def compute(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> HalfTrendState:
        """Compute HalfTrend signal from OHLC arrays."""
        if len(closes) < max(self._amplitude, self._atr_period) + 1:
            return HalfTrendState()

        atr = self._compute_atr(highs, lows, closes, self._atr_period)

        # Upper / lower bands using recent amplitude-window extremes
        amp = self._amplitude
        recent_high = float(np.max(highs[-amp:]))
        recent_low = float(np.min(lows[-amp:]))

        upper_band = recent_high - atr * 1.5
        lower_band = recent_low + atr * 1.5

        current_close = float(closes[-1])

        if current_close > upper_band:
            trend = "UP"
            signal = 1
        elif current_close < lower_band:
            trend = "DOWN"
            signal = -1
        else:
            trend = "FLAT"
            signal = 0

        # Strength: distance from relevant band normalized by ATR
        if trend == "UP":
            strength = abs(current_close - upper_band) / atr if atr > 0 else 0.0
        elif trend == "DOWN":
            strength = abs(current_close - lower_band) / atr if atr > 0 else 0.0
        else:
            strength = 0.0

        return HalfTrendState(
            signal=signal,
            upper_band=upper_band,
            lower_band=lower_band,
            strength=min(strength, 3.0),  # Cap at 3.0
            trend=trend,
            atr=atr,
        )

    @staticmethod
    def _compute_atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
    ) -> float:
        if len(closes) < 2:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    def to_dict(self, state: HalfTrendState) -> dict:
        return {
            "halftrend_signal": state.signal,
            "halftrend_upper": state.upper_band,
            "halftrend_lower": state.lower_band,
            "halftrend_strength": state.strength,
            "halftrend_trend": state.trend,
        }
