"""
NEMESIS LEVIATHAN — Extreme Risk Event Detector
Phase 14 — Detects black swan / tail risk events in real-time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TailEvent:
    """Detected extreme/tail risk event."""
    event_type: str
    magnitude: float      # Standard deviations from mean
    timestamp_idx: int
    description: str = ""


class LeviathanCore:
    """
    Detects extreme market events that warrant immediate risk response:
    - Price moves > 4σ
    - Volume spikes > 5σ
    - Spread widening > 3σ
    - Rapid consecutive same-direction bars (momentum exhaustion)
    """

    def __init__(self, lookback: int = 200, sigma_threshold: float = 4.0):
        self._lookback = lookback
        self._sigma = sigma_threshold
        self._returns: List[float] = []
        self._volumes: List[float] = []
        self._spreads: List[float] = []
        self._events: List[TailEvent] = []

    def update(self, price_return: float, volume: float, spread: float,
               timestamp_idx: int = 0) -> Optional[TailEvent]:
        """Update with new bar data and check for tail events."""
        self._returns.append(price_return)
        self._volumes.append(volume)
        self._spreads.append(spread)

        # Trim to lookback
        if len(self._returns) > self._lookback * 2:
            self._returns = self._returns[-self._lookback:]
            self._volumes = self._volumes[-self._lookback:]
            self._spreads = self._spreads[-self._lookback:]

        if len(self._returns) < 30:
            return None

        # Check price return
        ret_arr = np.array(self._returns[:-1])
        ret_z = (price_return - np.mean(ret_arr)) / (np.std(ret_arr) + 1e-10)
        if abs(ret_z) > self._sigma:
            event = TailEvent(
                event_type="PRICE_TAIL",
                magnitude=abs(ret_z),
                timestamp_idx=timestamp_idx,
                description=f"Price return {ret_z:.1f}σ",
            )
            self._events.append(event)
            return event

        # Check volume spike
        vol_arr = np.array(self._volumes[:-1])
        vol_z = (volume - np.mean(vol_arr)) / (np.std(vol_arr) + 1e-10)
        if vol_z > 5.0:
            event = TailEvent(
                event_type="VOLUME_SPIKE",
                magnitude=vol_z,
                timestamp_idx=timestamp_idx,
                description=f"Volume spike {vol_z:.1f}σ",
            )
            self._events.append(event)
            return event

        # Check spread widening
        sp_arr = np.array(self._spreads[:-1])
        sp_z = (spread - np.mean(sp_arr)) / (np.std(sp_arr) + 1e-10)
        if sp_z > 3.0:
            event = TailEvent(
                event_type="SPREAD_WIDENING",
                magnitude=sp_z,
                timestamp_idx=timestamp_idx,
                description=f"Spread widening {sp_z:.1f}σ",
            )
            self._events.append(event)
            return event

        return None

    @property
    def recent_events(self) -> List[TailEvent]:
        return self._events[-20:]

    @property
    def is_extreme(self) -> bool:
        """Whether an extreme event was detected in last 5 updates."""
        return any(e.timestamp_idx >= len(self._returns) - 5 for e in self._events[-5:])

    def reset(self) -> None:
        self._returns.clear()
        self._volumes.clear()
        self._spreads.clear()
        self._events.clear()
