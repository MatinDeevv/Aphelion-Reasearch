"""
APHELION Microstructure Features
VPIN, OFI, Tick Entropy, Hawkes Intensity, Spread Dynamics.
Section 5.1 of the Engineering Spec.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MicrostructureState:
    """Rolling state for microstructure computation."""
    vpin: float = 0.0
    ofi: float = 0.0
    tick_entropy: float = 0.0
    hawkes_buy_intensity: float = 0.0
    hawkes_sell_intensity: float = 0.0
    micro_price_divergence: float = 0.0
    bid_ask_spread: float = 0.0
    spread_velocity: float = 0.0
    quote_depth: float = 0.0


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading.
    High VPIN = informed trading in progress.
    """

    def __init__(self, bucket_size: int = 50, n_buckets: int = 50):
        self._bucket_size = bucket_size
        self._n_buckets = n_buckets
        self._current_bucket_volume = 0.0
        self._current_buy_volume = 0.0
        self._buckets: deque[tuple[float, float]] = deque(maxlen=n_buckets)
        self._last_price: float = 0.0

    def update(self, price: float, volume: float) -> float:
        """Update VPIN with new tick. Returns current VPIN."""
        if self._last_price == 0:
            self._last_price = price
            return 0.0

        # Classify volume as buy or sell using tick rule
        if price > self._last_price:
            buy_vol = volume
        elif price < self._last_price:
            buy_vol = 0.0
        else:
            buy_vol = volume * 0.5

        self._last_price = price
        self._current_bucket_volume += volume
        self._current_buy_volume += buy_vol

        # Check if bucket is complete
        if self._current_bucket_volume >= self._bucket_size:
            sell_vol = self._current_bucket_volume - self._current_buy_volume
            self._buckets.append((self._current_buy_volume, sell_vol))
            self._current_bucket_volume = 0.0
            self._current_buy_volume = 0.0

        return self._compute()

    def _compute(self) -> float:
        if len(self._buckets) < 2:
            return 0.0
        total_imbalance = sum(abs(b - s) for b, s in self._buckets)
        total_volume = sum(b + s for b, s in self._buckets)
        if total_volume == 0:
            return 0.0
        return total_imbalance / total_volume


class OFICalculator:
    """
    Order Flow Imbalance.
    Positive = buy pressure, Negative = sell pressure.
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._prev_bid: float = 0.0
        self._prev_ask: float = 0.0
        self._prev_bid_size: float = 0.0
        self._prev_ask_size: float = 0.0
        self._ofi_values: deque[float] = deque(maxlen=window)

    def update(self, bid: float, ask: float,
               bid_size: float = 1.0, ask_size: float = 1.0) -> float:
        """Compute OFI from bid/ask changes."""
        if self._prev_bid == 0:
            self._prev_bid = bid
            self._prev_ask = ask
            self._prev_bid_size = bid_size
            self._prev_ask_size = ask_size
            return 0.0

        # Bid delta
        if bid > self._prev_bid:
            bid_delta = bid_size
        elif bid == self._prev_bid:
            bid_delta = bid_size - self._prev_bid_size
        else:
            bid_delta = -self._prev_bid_size

        # Ask delta
        if ask < self._prev_ask:
            ask_delta = -ask_size
        elif ask == self._prev_ask:
            ask_delta = -(ask_size - self._prev_ask_size)
        else:
            ask_delta = self._prev_ask_size

        ofi = bid_delta + ask_delta
        self._ofi_values.append(ofi)

        self._prev_bid = bid
        self._prev_ask = ask
        self._prev_bid_size = bid_size
        self._prev_ask_size = ask_size

        return sum(self._ofi_values)


class TickEntropyCalculator:
    """
    Shannon entropy of tick direction sequence.
    Low entropy = directional movement, High entropy = noise.
    """

    def __init__(self, window: int = 100):
        self._window = window
        self._directions: deque[int] = deque(maxlen=window)  # +1 or -1
        self._last_price: float = 0.0

    def update(self, price: float) -> float:
        if self._last_price == 0:
            self._last_price = price
            return 1.0  # Maximum entropy initially

        direction = 1 if price >= self._last_price else -1
        self._directions.append(direction)
        self._last_price = price

        if len(self._directions) < 10:
            return 1.0

        return self._compute()

    def _compute(self) -> float:
        n = len(self._directions)
        ups = sum(1 for d in self._directions if d == 1)
        downs = n - ups

        if ups == 0 or downs == 0:
            return 0.0  # Perfect directionality

        p_up = ups / n
        p_down = downs / n

        entropy = -(p_up * np.log2(p_up) + p_down * np.log2(p_down))
        return entropy


class HawkesIntensity:
    """
    Self-exciting Hawkes process for buy/sell order arrival.
    Accelerating intensity = accumulation/distribution.
    """

    def __init__(self, decay: float = 0.1, baseline: float = 1.0):
        self._decay = decay
        self._baseline = baseline
        self._events: deque[float] = deque(maxlen=1000)
        self._intensity = baseline

    def update(self, timestamp: float) -> float:
        """Record an event and return current intensity."""
        self._events.append(timestamp)
        self._intensity = self._compute(timestamp)
        return self._intensity

    def _compute(self, t: float) -> float:
        intensity = self._baseline
        for event_time in self._events:
            dt = t - event_time
            if dt > 0:
                intensity += np.exp(-self._decay * dt)
        return intensity

    @property
    def intensity(self) -> float:
        return self._intensity


class MicrostructureEngine:
    """Computes all microstructure features from raw tick data."""

    def __init__(self, vpin_bucket_size: int = 50, ofi_window: int = 50,
                 entropy_window: int = 100, hawkes_decay: float = 0.1):
        self._vpin = VPINCalculator(bucket_size=vpin_bucket_size)
        self._ofi = OFICalculator(window=ofi_window)
        self._entropy = TickEntropyCalculator(window=entropy_window)
        self._hawkes_buy = HawkesIntensity(decay=hawkes_decay)
        self._hawkes_sell = HawkesIntensity(decay=hawkes_decay)
        self._last_spread: float = 0.0
        self._spread_history: deque[float] = deque(maxlen=50)
        self._state = MicrostructureState()

    def update(self, timestamp: float, bid: float, ask: float,
               last_price: float, volume: float,
               bid_size: float = 1.0, ask_size: float = 1.0) -> MicrostructureState:
        """Update all microstructure features with new tick data."""
        # VPIN
        self._state.vpin = self._vpin.update(last_price, volume)

        # OFI
        self._state.ofi = self._ofi.update(bid, ask, bid_size, ask_size)

        # Tick Entropy
        self._state.tick_entropy = self._entropy.update(last_price)

        # Hawkes Intensity — classify tick as buy or sell
        mid = (bid + ask) / 2.0
        if last_price >= mid:
            self._state.hawkes_buy_intensity = self._hawkes_buy.update(timestamp)
        else:
            self._state.hawkes_sell_intensity = self._hawkes_sell.update(timestamp)

        # Micro-price divergence
        if bid_size + ask_size > 0:
            weighted_mid = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
            self._state.micro_price_divergence = mid - weighted_mid
        else:
            self._state.micro_price_divergence = 0.0

        # Spread dynamics
        spread = ask - bid
        self._state.bid_ask_spread = spread
        self._spread_history.append(spread)
        if len(self._spread_history) >= 2:
            self._state.spread_velocity = spread - self._spread_history[-2]
        else:
            self._state.spread_velocity = 0.0

        return self._state

    def to_dict(self) -> dict:
        return {
            "vpin": self._state.vpin,
            "ofi": self._state.ofi,
            "tick_entropy": self._state.tick_entropy,
            "hawkes_buy_intensity": self._state.hawkes_buy_intensity,
            "hawkes_sell_intensity": self._state.hawkes_sell_intensity,
            "micro_price_divergence": self._state.micro_price_divergence,
            "bid_ask_spread": self._state.bid_ask_spread,
            "spread_velocity": self._state.spread_velocity,
        }
