"""
FLOW Order Flow Analysis
Tick-level delta computation and trade classification.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class OrderFlowState:
    delta: float = 0.0               # Buy - Sell volume ratio [-1, 1]
    cumulative_delta: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    aggression_ratio: float = 0.0    # Aggressive buyer/seller imbalance


class OrderFlowAnalyzer:
    """
    Analyzes tick-level order flow to detect institutional activity.
    Uses tick rule for trade classification when bid/ask not available.
    """

    def __init__(self, window: int = 100):
        self._window = window
        self._deltas: List[float] = []
        self._cumulative_delta = 0.0
        self._last_price = 0.0

    def update_tick(self, price: float, volume: float) -> OrderFlowState:
        """Process a single tick and update flow state."""
        if self._last_price == 0:
            self._last_price = price
            return OrderFlowState()

        # Tick rule classification
        if price > self._last_price:
            buy_vol = volume
            sell_vol = 0.0
        elif price < self._last_price:
            buy_vol = 0.0
            sell_vol = volume
        else:
            # Price unchanged — split 50/50
            buy_vol = volume * 0.5
            sell_vol = volume * 0.5

        self._last_price = price

        total = buy_vol + sell_vol
        delta = (buy_vol - sell_vol) / total if total > 0 else 0.0
        self._deltas.append(delta)
        self._cumulative_delta += delta

        # Keep window
        if len(self._deltas) > self._window:
            self._deltas = self._deltas[-self._window:]

        return self._compute_state(buy_vol, sell_vol)

    def compute_bar_delta(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> OrderFlowState:
        """Compute delta from an array of intra-bar prices and volumes."""
        if len(prices) < 2:
            return OrderFlowState()

        buy_vol = 0.0
        sell_vol = 0.0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                buy_vol += volumes[i]
            elif prices[i] < prices[i - 1]:
                sell_vol += volumes[i]
            else:
                buy_vol += volumes[i] * 0.5
                sell_vol += volumes[i] * 0.5

        total = buy_vol + sell_vol
        delta = (buy_vol - sell_vol) / total if total > 0 else 0.0
        aggression = abs(delta)

        return OrderFlowState(
            delta=delta,
            cumulative_delta=self._cumulative_delta + delta,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            total_volume=total,
            aggression_ratio=aggression,
        )

    def _compute_state(self, buy_vol: float, sell_vol: float) -> OrderFlowState:
        total = buy_vol + sell_vol
        recent_delta = np.mean(self._deltas[-20:]) if self._deltas else 0.0
        return OrderFlowState(
            delta=float(recent_delta),
            cumulative_delta=self._cumulative_delta,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            total_volume=total,
            aggression_ratio=abs(float(recent_delta)),
        )

    def reset(self) -> None:
        self._deltas.clear()
        self._cumulative_delta = 0.0
        self._last_price = 0.0
