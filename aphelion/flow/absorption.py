"""
FLOW Volume Absorption Detection
Detects large volume being absorbed at a level without price movement.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np


@dataclass
class AbsorptionEvent:
    price: float
    volume_absorbed: float
    direction: str           # "BUY_ABSORBED" or "SELL_ABSORBED"
    strength: float          # [0, 1]
    bar_count: int           # How many bars the absorption lasted


class AbsorptionDetector:
    """
    Detects volume absorption: large volume at a price level with minimal price change.
    This indicates institutional accumulation or distribution.
    """

    def __init__(self, volume_threshold_z: float = 2.0, price_threshold_pct: float = 0.05,
                 lookback: int = 50):
        self._vol_z_threshold = volume_threshold_z
        self._price_pct_threshold = price_threshold_pct
        self._lookback = lookback
        self._volume_history: deque = deque(maxlen=lookback)
        self._absorption_events: list = []

    def update(
        self, high: float, low: float, close: float, volume: float, open_price: float
    ) -> Optional[AbsorptionEvent]:
        """Check for absorption on this bar."""
        self._volume_history.append(volume)

        if len(self._volume_history) < 10:
            return None

        vol_arr = np.array(self._volume_history)
        vol_mean = float(np.mean(vol_arr))
        vol_std = float(np.std(vol_arr))

        if vol_std == 0:
            return None

        vol_z = (volume - vol_mean) / vol_std
        price_range_pct = abs(close - open_price) / open_price * 100 if open_price > 0 else 0

        # High volume + low price movement = absorption
        if vol_z > self._vol_z_threshold and price_range_pct < self._price_pct_threshold:
            direction = "BUY_ABSORBED" if close < open_price else "SELL_ABSORBED"
            strength = min(1.0, vol_z / 4.0)

            event = AbsorptionEvent(
                price=close,
                volume_absorbed=volume,
                direction=direction,
                strength=strength,
                bar_count=1,
            )
            self._absorption_events.append(event)
            return event

        return None

    @property
    def recent_events(self) -> list:
        return self._absorption_events[-10:]

    def reset(self) -> None:
        self._volume_history.clear()
        self._absorption_events.clear()
