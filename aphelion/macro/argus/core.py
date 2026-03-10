"""
ARGUS — Market Surveillance Core
Phase 19 — Engineering Spec v3.0

Monitors market microstructure for anomalies: stale data, flash crashes,
market halts, and unusual price patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketAnomaly:
    """Detected market anomaly."""
    anomaly_type: str       # STALE_FEED, FLASH_MOVE, SPREAD_BLOWOUT, VOLUME_ANOMALY
    severity: float         # [0, 1]
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ArgusCore:
    """
    Market surveillance engine.
    Detects microstructure anomalies in real-time.
    """

    def __init__(
        self,
        stale_threshold_sec: float = 5.0,
        flash_atr_mult: float = 3.0,
        spread_mult: float = 5.0,
    ):
        self._stale_thresh = stale_threshold_sec
        self._flash_mult = flash_atr_mult
        self._spread_mult = spread_mult
        self._last_tick_time: Optional[datetime] = None
        self._anomalies: List[MarketAnomaly] = []
        self._normal_spread: float = 0.0

    def set_normal_spread(self, spread: float) -> None:
        self._normal_spread = spread

    def on_tick(
        self,
        price: float,
        prev_price: float,
        spread: float,
        volume: float,
        avg_volume: float,
        atr: float,
        now: Optional[datetime] = None,
    ) -> List[MarketAnomaly]:
        """Process a tick and detect anomalies."""
        now = now or datetime.now(timezone.utc)
        detected = []

        # Stale feed detection
        if self._last_tick_time is not None:
            gap = (now - self._last_tick_time).total_seconds()
            if gap > self._stale_thresh:
                a = MarketAnomaly("STALE_FEED", min(1.0, gap / 30.0), f"Feed gap: {gap:.1f}s")
                detected.append(a)
        self._last_tick_time = now

        # Flash move
        if atr > 0:
            move = abs(price - prev_price)
            if move > atr * self._flash_mult:
                sev = min(1.0, move / (atr * self._flash_mult * 2))
                detected.append(MarketAnomaly("FLASH_MOVE", sev, f"Move {move:.2f} > {self._flash_mult}x ATR"))

        # Spread blowout
        if self._normal_spread > 0 and spread > self._normal_spread * self._spread_mult:
            sev = min(1.0, spread / (self._normal_spread * self._spread_mult * 2))
            detected.append(MarketAnomaly("SPREAD_BLOWOUT", sev, f"Spread {spread:.1f} > {self._spread_mult}x normal"))

        # Volume anomaly
        if avg_volume > 0 and volume > avg_volume * 10:
            sev = min(1.0, volume / (avg_volume * 20))
            detected.append(MarketAnomaly("VOLUME_ANOMALY", sev, f"Volume {volume:.0f} > 10x avg"))

        self._anomalies.extend(detected)
        return detected

    @property
    def recent_anomalies(self) -> List[MarketAnomaly]:
        return self._anomalies[-50:]

    @property
    def alert_active(self) -> bool:
        if not self._anomalies:
            return False
        recent = self._anomalies[-5:]
        return any(a.severity > 0.7 for a in recent)
