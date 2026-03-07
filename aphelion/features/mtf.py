"""
APHELION Multi-Timeframe Alignment
Computes alignment scores across M1, M5, M15, H1.
"""

import numpy as np
import pandas as pd
from typing import Optional

from aphelion.core.config import Timeframe, TIMEFRAMES


class MTFAlignmentEngine:
    """
    Multi-timeframe alignment scoring.
    Checks if signals agree across all 4 timeframes.
    """

    def __init__(self):
        self._signals: dict[Timeframe, str] = {}  # LONG/SHORT/FLAT
        self._weights: dict[Timeframe, float] = {
            Timeframe.M1: 0.15,
            Timeframe.M5: 0.25,
            Timeframe.M15: 0.30,
            Timeframe.H1: 0.30,
        }

    def set_weights(self, weights: dict[Timeframe, float]) -> None:
        """Set dynamic weights from MERIDIAN causality analysis."""
        total = sum(weights.values())
        if total > 0:
            self._weights = {tf: w / total for tf, w in weights.items()}

    def compute_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        """Determine trend direction from price data."""
        if len(df) < period:
            return "FLAT"

        closes = df["close"].values
        sma = np.mean(closes[-period:])
        current = closes[-1]

        slope = np.polyfit(range(period), closes[-period:], 1)[0]

        if current > sma and slope > 0:
            return "LONG"
        elif current < sma and slope < 0:
            return "SHORT"
        return "FLAT"

    def update_signal(self, timeframe: Timeframe, signal: str) -> None:
        """Update the signal for a specific timeframe."""
        self._signals[timeframe] = signal

    def alignment_count(self, direction: Optional[str] = None) -> int:
        """Count how many timeframes agree on a direction."""
        if not self._signals:
            return 0

        if direction is None:
            # Count the most common non-FLAT direction
            longs = sum(1 for s in self._signals.values() if s == "LONG")
            shorts = sum(1 for s in self._signals.values() if s == "SHORT")
            return max(longs, shorts)

        return sum(1 for s in self._signals.values() if s == direction)

    def weighted_alignment(self, direction: str) -> float:
        """Compute weighted alignment score for a direction."""
        score = 0.0
        for tf, signal in self._signals.items():
            if signal == direction:
                score += self._weights.get(tf, 0)
        return score

    def compute(self, bars_by_tf: dict[Timeframe, pd.DataFrame],
                period: int = 20) -> dict:
        """Compute MTF alignment from bar data across timeframes."""
        for tf in TIMEFRAMES:
            if tf in bars_by_tf and len(bars_by_tf[tf]) >= period:
                trend = self.compute_trend(bars_by_tf[tf], period)
                self.update_signal(tf, trend)

        # Determine dominant direction
        longs = self.alignment_count("LONG")
        shorts = self.alignment_count("SHORT")

        if longs > shorts:
            dominant = "LONG"
        elif shorts > longs:
            dominant = "SHORT"
        else:
            dominant = "FLAT"

        return {
            "mtf_alignment_count": max(longs, shorts),
            "mtf_dominant_direction": dominant,
            "mtf_long_count": longs,
            "mtf_short_count": shorts,
            "mtf_weighted_score": self.weighted_alignment(dominant) if dominant != "FLAT" else 0.0,
            "mtf_signals": {tf.value: sig for tf, sig in self._signals.items()},
        }
