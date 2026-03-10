"""
FLOW Stop Hunt (Liquidity Sweep) Detection
Detects when price spikes through a liquidity zone and reverses.
HIGH-QUALITY contrarian signal.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from aphelion.flow.liquidity import LiquidityZone


@dataclass
class StopHuntSignal:
    direction: int           # 1=BUY (swept below, reversed up), -1=SELL
    zone: LiquidityZone
    sweep_magnitude: float   # How far price went past the zone
    confidence: float        # [0, 1]
    bar_index: int = -1


class StopHuntDetector:
    """
    A liquidity sweep (stop hunt) pattern:
    1. Price spikes above/below a known liquidity zone
    2. Volume spikes (institutions taking the stops)
    3. Price quickly reverses back inside the zone

    This is a HIGH-QUALITY trading signal for the reversal.
    """

    def __init__(self, volume_spike_threshold: float = 1.5):
        self._vol_spike_threshold = volume_spike_threshold

    def detect(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        zones: List[LiquidityZone],
    ) -> Optional[StopHuntSignal]:
        """Check if the most recent bar represents a stop hunt reversal."""
        if len(closes) < 3:
            return None

        current_high = float(highs[-1])
        current_low = float(lows[-1])
        current_close = float(closes[-1])
        prev_close = float(closes[-2])
        current_volume = float(volumes[-1])

        # Mean volume for reference
        mean_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        vol_ratio = current_volume / mean_vol if mean_vol > 0 else 1.0

        for zone in zones:
            # Check sweep above resistance
            swept_above = prev_close < zone.price < current_high
            # Check sweep below support
            swept_below = prev_close > zone.price > current_low

            if swept_above and current_close < zone.price:
                # Swept above + closed below → bearish reversal
                magnitude = current_high - zone.price
                confidence = self._compute_confidence(vol_ratio, magnitude, zone.strength)
                return StopHuntSignal(
                    direction=-1,
                    zone=zone,
                    sweep_magnitude=magnitude,
                    confidence=confidence,
                )

            if swept_below and current_close > zone.price:
                # Swept below + closed above → bullish reversal
                magnitude = zone.price - current_low
                confidence = self._compute_confidence(vol_ratio, magnitude, zone.strength)
                return StopHuntSignal(
                    direction=1,
                    zone=zone,
                    sweep_magnitude=magnitude,
                    confidence=confidence,
                )

        return None

    def _compute_confidence(
        self, vol_ratio: float, sweep_magnitude: float, zone_strength: float
    ) -> float:
        """Compute stop hunt confidence from contributing factors."""
        vol_score = min(1.0, vol_ratio / 3.0)
        magnitude_score = min(1.0, sweep_magnitude / 10.0)  # 10 pips = max score
        base = vol_score * 0.4 + magnitude_score * 0.3 + zone_strength * 0.3
        return min(1.0, max(0.0, base))
