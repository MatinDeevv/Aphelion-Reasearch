"""
PHANTOM — Hidden Liquidity Detector Core
Phase 18 — Engineering Spec v3.0

Detects hidden (iceberg) orders and dark pool prints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HiddenOrder:
    """A detected hidden order signal."""
    price: float
    estimated_size: float
    direction: int       # 1 = bid, -1 = ask
    confidence: float    # [0, 1]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PhantomCore:
    """
    Detects hidden liquidity via:
    1. Repeated fills at same level without visible size
    2. Abnormal trade-through patterns
    3. Volume inconsistency (large executed volume vs small visible book)
    """

    def __init__(self, min_fill_count: int = 3, size_ratio_threshold: float = 3.0):
        self._min_fills = min_fill_count
        self._size_ratio = size_ratio_threshold
        self._fill_tracker: dict = {}  # price -> fill_count
        self._detections: List[HiddenOrder] = []

    def on_trade(
        self,
        price: float,
        volume: float,
        visible_size: float,
        side: int,
    ) -> Optional[HiddenOrder]:
        """Process a trade and detect hidden order patterns."""
        # Track fills at this price level
        key = round(price, 2)
        self._fill_tracker.setdefault(key, {"count": 0, "total_volume": 0.0, "side": side})
        self._fill_tracker[key]["count"] += 1
        self._fill_tracker[key]["total_volume"] += volume

        # Check for iceberg pattern
        entry = self._fill_tracker[key]
        if (
            entry["count"] >= self._min_fills
            and visible_size > 0
            and entry["total_volume"] / visible_size > self._size_ratio
        ):
            confidence = min(1.0, entry["count"] / (self._min_fills * 3))
            detection = HiddenOrder(
                price=price,
                estimated_size=entry["total_volume"],
                direction=side,
                confidence=confidence,
            )
            self._detections.append(detection)
            return detection

        return None

    def reset_level(self, price: float) -> None:
        key = round(price, 2)
        self._fill_tracker.pop(key, None)

    @property
    def recent_detections(self) -> List[HiddenOrder]:
        return self._detections[-20:]

    @property
    def active_levels(self) -> int:
        return len(self._fill_tracker)
