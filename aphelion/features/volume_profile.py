"""
APHELION Volume Profile Features
Volume Delta, CVD, POC, VAH/VAL, Delta Divergence, Absorption.
Section 5.3 of the Engineering Spec.
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass


@dataclass
class VolumeProfileState:
    volume_delta: float = 0.0
    cumulative_delta: float = 0.0
    poc: float = 0.0           # Point of Control
    vah: float = 0.0           # Value Area High
    val: float = 0.0           # Value Area Low
    delta_divergence: bool = False
    absorption: bool = False


class VolumeDeltaCalculator:
    """
    Aggressive buy volume minus aggressive sell volume per bar.
    Uses tick rule: uptick = buy, downtick = sell.
    """

    def __init__(self):
        self._cumulative = 0.0
        self._deltas: deque[float] = deque(maxlen=500)
        self._prices: deque[float] = deque(maxlen=500)

    def compute_bar_delta(self, open_price: float, high: float, low: float,
                          close: float, volume: float) -> float:
        """
        Estimate volume delta from OHLCV using the close-open heuristic.
        Buy volume = volume * (close - low) / (high - low)
        """
        bar_range = high - low
        if bar_range == 0 or volume == 0:
            return 0.0

        buy_fraction = (close - low) / bar_range
        buy_vol = volume * buy_fraction
        sell_vol = volume * (1 - buy_fraction)
        delta = buy_vol - sell_vol

        self._deltas.append(delta)
        self._prices.append(close)
        self._cumulative += delta
        return delta

    @property
    def cumulative(self) -> float:
        return self._cumulative

    def reset_session(self) -> None:
        self._cumulative = 0.0

    def detect_divergence(self, price_highs: np.ndarray, n: int = 20) -> bool:
        """
        Delta divergence: price making new high while CVD making lower high.
        Classic distribution signal.
        """
        if len(self._prices) < n or len(self._deltas) < n:
            return False

        prices = np.array(list(self._prices)[-n:])
        deltas = np.array(list(self._deltas)[-n:])
        cvd = np.cumsum(deltas)

        # Check if price is at recent high but CVD is not
        price_at_high = prices[-1] >= np.max(prices[:-1]) * 0.999
        cvd_below_high = cvd[-1] < np.max(cvd[:-1]) * 0.95

        return price_at_high and cvd_below_high


class VolumeProfileCalculator:
    """
    Session volume profile: POC, VAH, VAL.
    """

    def __init__(self, n_bins: int = 50, value_area_pct: float = 0.70):
        self._n_bins = n_bins
        self._value_area_pct = value_area_pct

    def compute(self, highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, volumes: np.ndarray) -> dict:
        """Compute volume profile from bar data."""
        if len(closes) == 0:
            return {"poc": 0.0, "vah": 0.0, "val": 0.0}

        price_min = np.min(lows)
        price_max = np.max(highs)

        if price_max == price_min:
            return {"poc": closes[-1], "vah": closes[-1], "val": closes[-1]}

        # Create price bins
        bin_edges = np.linspace(price_min, price_max, self._n_bins + 1)
        bin_volumes = np.zeros(self._n_bins)

        # Distribute each bar's volume across price bins it spans
        for i in range(len(closes)):
            bar_low = lows[i]
            bar_high = highs[i]
            bar_vol = volumes[i] if volumes[i] > 0 else 1.0

            for j in range(self._n_bins):
                bin_low = bin_edges[j]
                bin_high = bin_edges[j + 1]

                # Overlap between bar range and bin
                overlap_low = max(bar_low, bin_low)
                overlap_high = min(bar_high, bin_high)

                if overlap_high > overlap_low:
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        fraction = (overlap_high - overlap_low) / bar_range
                        bin_volumes[j] += bar_vol * fraction

        # POC: price level with highest volume
        poc_bin = np.argmax(bin_volumes)
        poc = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

        # Value Area: 70% of volume centered on POC
        total_vol = np.sum(bin_volumes)
        if total_vol == 0:
            return {"poc": poc, "vah": price_max, "val": price_min}

        target_vol = total_vol * self._value_area_pct
        accumulated = bin_volumes[poc_bin]
        low_idx = poc_bin
        high_idx = poc_bin

        while accumulated < target_vol:
            expand_up = bin_volumes[high_idx + 1] if high_idx + 1 < self._n_bins else 0
            expand_down = bin_volumes[low_idx - 1] if low_idx > 0 else 0

            if expand_up >= expand_down and high_idx + 1 < self._n_bins:
                high_idx += 1
                accumulated += bin_volumes[high_idx]
            elif low_idx > 0:
                low_idx -= 1
                accumulated += bin_volumes[low_idx]
            else:
                break

        val = bin_edges[low_idx]
        vah = bin_edges[high_idx + 1]

        return {"poc": poc, "vah": vah, "val": val}


class AbsorptionDetector:
    """
    Detects absorption: large volume with minimal price movement.
    Institutional absorption pattern.
    """

    def __init__(self, volume_threshold_mult: float = 2.0,
                 price_move_threshold: float = 0.3):
        self._vol_mult = volume_threshold_mult
        self._price_threshold = price_move_threshold
        self._avg_volumes: deque[float] = deque(maxlen=50)
        self._avg_ranges: deque[float] = deque(maxlen=50)

    def check(self, high: float, low: float, close: float,
              open_price: float, volume: float) -> bool:
        """Check if current bar shows absorption."""
        bar_range = high - low
        self._avg_volumes.append(volume)
        self._avg_ranges.append(bar_range)

        if len(self._avg_volumes) < 20:
            return False

        avg_vol = np.mean(list(self._avg_volumes)[:-1])
        avg_range = np.mean(list(self._avg_ranges)[:-1])

        if avg_vol == 0 or avg_range == 0:
            return False

        vol_ratio = volume / avg_vol
        range_ratio = bar_range / avg_range

        # High volume + small range = absorption
        return vol_ratio > self._vol_mult and range_ratio < self._price_threshold


class VolumeProfileEngine:
    """Computes all volume profile features."""

    def __init__(self):
        self._delta = VolumeDeltaCalculator()
        self._profile = VolumeProfileCalculator()
        self._absorption = AbsorptionDetector()
        self._state = VolumeProfileState()

    def update_bar(self, open_price: float, high: float, low: float,
                   close: float, volume: float) -> VolumeProfileState:
        """Update with new bar data."""
        self._state.volume_delta = self._delta.compute_bar_delta(
            open_price, high, low, close, volume
        )
        self._state.cumulative_delta = self._delta.cumulative
        self._state.absorption = self._absorption.check(
            high, low, close, open_price, volume
        )
        return self._state

    def compute_session_profile(self, df: pd.DataFrame) -> dict:
        """Compute full session volume profile."""
        if df.empty:
            return {"poc": 0, "vah": 0, "val": 0}

        result = self._profile.compute(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values if "volume" in df.columns else np.ones(len(df)),
        )

        self._state.poc = result["poc"]
        self._state.vah = result["vah"]
        self._state.val = result["val"]
        return result

    def check_divergence(self, price_highs: np.ndarray) -> bool:
        self._state.delta_divergence = self._delta.detect_divergence(price_highs)
        return self._state.delta_divergence

    def reset_session(self) -> None:
        self._delta.reset_session()

    def to_dict(self) -> dict:
        return {
            "volume_delta": self._state.volume_delta,
            "cumulative_delta": self._state.cumulative_delta,
            "poc": self._state.poc,
            "vah": self._state.vah,
            "val": self._state.val,
            "delta_divergence": self._state.delta_divergence,
            "absorption": self._state.absorption,
        }
