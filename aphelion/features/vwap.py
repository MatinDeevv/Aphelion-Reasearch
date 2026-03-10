"""
APHELION VWAP Features
Session VWAP, Anchored VWAP, Rolling VWAP with standard deviation bands.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class VWAPState:
    session_vwap: float = 0.0
    anchored_vwap: float = 0.0
    rolling_vwap: float = 0.0
    upper_band_1: float = 0.0  # +1 std dev
    lower_band_1: float = 0.0  # -1 std dev
    upper_band_2: float = 0.0  # +2 std dev
    lower_band_2: float = 0.0  # -2 std dev
    price_vs_vwap: float = 0.0  # Current price relative to VWAP


class VWAPCalculator:
    """Compute VWAP variants with standard deviation bands."""

    def __init__(self, rolling_window: int = 100):
        self._rolling_window = rolling_window
        # Session VWAP accumulators
        self._session_cum_tp_vol = 0.0
        self._session_cum_vol = 0.0
        self._session_cum_tp2_vol = 0.0
        self._session_open_date = None  # Track session date for auto-reset
        # Anchored VWAP
        self._anchor_cum_tp_vol = 0.0
        self._anchor_cum_vol = 0.0
        self._anchor_cum_tp2_vol = 0.0
        # Rolling
        self._rolling_tp_vol: deque[float] = deque(maxlen=rolling_window)
        self._rolling_vol: deque[float] = deque(maxlen=rolling_window)
        self._rolling_tp2_vol: deque[float] = deque(maxlen=rolling_window)
        self._state = VWAPState()

    def update(self, high: float, low: float, close: float,
               volume: float, timestamp=None) -> VWAPState:
        """Update VWAP with new bar data.
        
        Args:
            timestamp: Optional datetime for auto session reset on new trading day (00:00 UTC).
        """
        # BUGFIX: Auto-reset session VWAP on new trading day
        if timestamp is not None:
            from datetime import datetime
            if hasattr(timestamp, 'date'):
                session_date = timestamp.date()
            else:
                session_date = datetime.utcfromtimestamp(timestamp).date() if isinstance(timestamp, (int, float)) else None
            
            if session_date and (self._session_open_date is None or session_date != self._session_open_date):
                self._session_open_date = session_date
                self.reset_session()

        typical_price = (high + low + close) / 3.0
        tp_vol = typical_price * volume
        tp2_vol = (typical_price ** 2) * volume

        # Session VWAP
        self._session_cum_tp_vol += tp_vol
        self._session_cum_vol += volume
        self._session_cum_tp2_vol += tp2_vol

        if self._session_cum_vol > 0:
            self._state.session_vwap = self._session_cum_tp_vol / self._session_cum_vol
            variance = (self._session_cum_tp2_vol / self._session_cum_vol) - \
                       (self._state.session_vwap ** 2)
            std = np.sqrt(max(0, variance))
            self._state.upper_band_1 = self._state.session_vwap + std
            self._state.lower_band_1 = self._state.session_vwap - std
            self._state.upper_band_2 = self._state.session_vwap + 2 * std
            self._state.lower_band_2 = self._state.session_vwap - 2 * std

        # Anchored VWAP
        self._anchor_cum_tp_vol += tp_vol
        self._anchor_cum_vol += volume
        if self._anchor_cum_vol > 0:
            self._state.anchored_vwap = self._anchor_cum_tp_vol / self._anchor_cum_vol

        # Rolling VWAP
        self._rolling_tp_vol.append(tp_vol)
        self._rolling_vol.append(volume)
        self._rolling_tp2_vol.append(tp2_vol)

        total_tp_vol = sum(self._rolling_tp_vol)
        total_vol = sum(self._rolling_vol)
        if total_vol > 0:
            self._state.rolling_vwap = total_tp_vol / total_vol

        # Price relative to VWAP
        self._state.price_vs_vwap = close - self._state.session_vwap

        return self._state

    def reset_session(self) -> None:
        """Reset session VWAP (called at session open)."""
        self._session_cum_tp_vol = 0.0
        self._session_cum_vol = 0.0
        self._session_cum_tp2_vol = 0.0
        self._state.session_vwap = 0.0
        self._state.upper_band_1 = 0.0
        self._state.lower_band_1 = 0.0
        self._state.upper_band_2 = 0.0
        self._state.lower_band_2 = 0.0
        self._state.price_vs_vwap = 0.0

    def set_anchor(self) -> None:
        """Set anchor point for anchored VWAP (e.g., at last swing)."""
        self._anchor_cum_tp_vol = 0.0
        self._anchor_cum_vol = 0.0
        self._anchor_cum_tp2_vol = 0.0

    def to_dict(self) -> dict:
        return {
            "session_vwap": self._state.session_vwap,
            "anchored_vwap": self._state.anchored_vwap,
            "rolling_vwap": self._state.rolling_vwap,
            "vwap_upper_1": self._state.upper_band_1,
            "vwap_lower_1": self._state.lower_band_1,
            "vwap_upper_2": self._state.upper_band_2,
            "vwap_lower_2": self._state.lower_band_2,
            "price_vs_vwap": self._state.price_vs_vwap,
        }
