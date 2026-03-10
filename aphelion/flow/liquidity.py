"""
FLOW Liquidity Zone Detection
Identifies institutional liquidity levels: prev-day H/L, round numbers, swing zones.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class LiquidityZone:
    """A price level where institutional orders accumulate."""
    price: float
    zone_type: str      # "SUPPORT", "RESISTANCE", "ROUND_NUMBER", "SWING"
    strength: float     # [0, 1] — higher = more significant
    touches: int = 0    # Number of times price tested this level
    source: str = ""    # Where the zone was identified from


class LiquidityZoneDetector:
    """
    XAU/USD liquidity zones are price levels where large institutional
    orders accumulate. Identifiable by:
    - Previous day high/low
    - Round numbers (e.g., 3100.00, 3150.00)
    - Recent swing highs/lows with multiple touches
    - Volume POC (Point of Control)
    """

    def __init__(self, round_number_spacing: float = 25.0, swing_lookback: int = 50,
                 min_touches: int = 3, touch_tolerance_pips: float = 5.0):
        self._spacing = round_number_spacing
        self._swing_lookback = swing_lookback
        self._min_touches = min_touches
        self._touch_tol = touch_tolerance_pips

    def detect_zones(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> List[LiquidityZone]:
        """Detect all active liquidity zones from recent bar data."""
        zones: List[LiquidityZone] = []

        if len(closes) < 2:
            return zones

        # 1. Previous day H/L (approximate: last 1440 M1 bars = 1 day)
        zones.extend(self._detect_prev_day_levels(highs, lows))

        # 2. Round number zones near current price
        zones.extend(self._detect_round_numbers(float(closes[-1])))

        # 3. Swing high/low zones with multiple touches
        zones.extend(self._detect_swing_zones(highs, lows, closes))

        # 4. Volume POC if volume data available
        if volumes is not None and len(volumes) > 20:
            poc = self._compute_volume_poc(closes, volumes)
            if poc is not None:
                zones.append(LiquidityZone(
                    price=poc, zone_type="RESISTANCE", strength=0.7,
                    source="volume_poc"
                ))

        # Sort by strength descending
        return sorted(zones, key=lambda z: z.strength, reverse=True)

    def _detect_prev_day_levels(
        self, highs: np.ndarray, lows: np.ndarray
    ) -> List[LiquidityZone]:
        zones = []
        # Use the last 1440 bars as "previous day" approximation
        lookback = min(1440, len(highs) - 1)
        if lookback < 10:
            return zones

        prev_highs = highs[-lookback - 1:-1]
        prev_lows = lows[-lookback - 1:-1]

        prev_high = float(np.max(prev_highs))
        prev_low = float(np.min(prev_lows))

        zones.append(LiquidityZone(
            price=prev_high, zone_type="RESISTANCE", strength=0.8,
            source="prev_day_high"
        ))
        zones.append(LiquidityZone(
            price=prev_low, zone_type="SUPPORT", strength=0.8,
            source="prev_day_low"
        ))
        return zones

    def _detect_round_numbers(self, current_price: float) -> List[LiquidityZone]:
        zones = []
        base = round(current_price / self._spacing) * self._spacing
        for offset in [-2, -1, 0, 1, 2]:
            level = base + offset * self._spacing
            distance = abs(current_price - level)
            # Closer round numbers are more relevant
            strength = max(0.3, 0.7 - distance / (self._spacing * 3))
            zones.append(LiquidityZone(
                price=level, zone_type="ROUND_NUMBER", strength=strength,
                source="round_number"
            ))
        return zones

    def _detect_swing_zones(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> List[LiquidityZone]:
        zones = []
        lookback = min(self._swing_lookback, len(closes))
        if lookback < 5:
            return zones

        h = highs[-lookback:]
        l = lows[-lookback:]

        # Find swing highs (local maxima)
        swing_highs = []
        swing_lows = []
        for i in range(2, len(h) - 2):
            if h[i] > h[i - 1] and h[i] > h[i - 2] and h[i] > h[i + 1] and h[i] > h[i + 2]:
                swing_highs.append(float(h[i]))
            if l[i] < l[i - 1] and l[i] < l[i - 2] and l[i] < l[i + 1] and l[i] < l[i + 2]:
                swing_lows.append(float(l[i]))

        # Cluster swing levels and count touches
        for level_list, zone_type in [(swing_highs, "RESISTANCE"), (swing_lows, "SUPPORT")]:
            clusters = self._cluster_levels(level_list, self._touch_tol)
            for price, touches in clusters:
                if touches >= self._min_touches:
                    strength = min(1.0, 0.5 + touches * 0.1)
                    zones.append(LiquidityZone(
                        price=price, zone_type=zone_type, strength=strength,
                        touches=touches, source="swing_zone"
                    ))

        return zones

    @staticmethod
    def _cluster_levels(levels: List[float], tolerance: float) -> List[tuple]:
        """Cluster nearby price levels and count touches."""
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if level - current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                avg_price = sum(current_cluster) / len(current_cluster)
                clusters.append((avg_price, len(current_cluster)))
                current_cluster = [level]
        if current_cluster:
            avg_price = sum(current_cluster) / len(current_cluster)
            clusters.append((avg_price, len(current_cluster)))
        return clusters

    @staticmethod
    def _compute_volume_poc(closes: np.ndarray, volumes: np.ndarray) -> Optional[float]:
        """Find the Volume Point of Control (price with highest volume)."""
        if len(closes) < 10:
            return None
        # Bin prices and sum volume per bin
        n_bins = 50
        price_min = float(np.min(closes))
        price_max = float(np.max(closes))
        if price_max == price_min:
            return float(price_min)
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_volumes = np.zeros(n_bins)
        for price, vol in zip(closes, volumes):
            idx = int((price - price_min) / (price_max - price_min) * (n_bins - 1))
            idx = max(0, min(idx, n_bins - 1))
            bin_volumes[idx] += vol
        poc_bin = int(np.argmax(bin_volumes))
        return float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2)
