"""Tests for Microstructure improvements: 3-state entropy, z-score OFI."""

import pytest
import math
from aphelion.features.microstructure import TickEntropyCalculator, OFICalculator


class TestTickEntropy3State:
    def test_max_entropy_three_states(self):
        """With 3 equally likely states, max entropy is log2(3) ~ 1.585 bits."""
        calc = TickEntropyCalculator(window=90)
        # Feed equal mix of up, down, flat ticks
        p = 2850.0
        entropy = 1.0
        for i in range(90):
            if i % 3 == 0:
                p += 0.10  # UP
            elif i % 3 == 1:
                p -= 0.10  # DOWN
            # else: FLAT (same price)
            entropy = calc.update(p)
        # Should be close to log2(3) = 1.585
        assert entropy > 1.0
        assert entropy <= math.log2(3) + 0.1

    def test_pure_uptrend_low_entropy(self):
        calc = TickEntropyCalculator(window=30)
        entropy = 1.0
        for i in range(35):
            entropy = calc.update(2850.0 + i * 0.1)
        # All UP ticks -> entropy should be 0 or very low
        assert entropy < 0.3

    def test_pure_downtrend_low_entropy(self):
        calc = TickEntropyCalculator(window=30)
        entropy = 1.0
        for i in range(35):
            entropy = calc.update(2900.0 - i * 0.1)
        assert entropy < 0.3

    def test_all_flat_low_entropy(self):
        calc = TickEntropyCalculator(window=30)
        entropy = 1.0
        for _ in range(35):
            entropy = calc.update(2850.0)
        assert entropy < 0.3


class TestOFIZScore:
    def test_normalized_returns_bounded(self):
        calc = OFICalculator(window=20)
        # Feed synthetic bid/ask data
        for i in range(30):
            bid = 2850.0 + (i % 5) * 0.01
            ask = bid + 0.03
            bid_vol = 100.0 + (i % 3) * 10
            ask_vol = 100.0 + ((i + 1) % 3) * 10
            calc.update(bid, ask, bid_vol, ask_vol)
        # Z-score normalization should clamp to [-1, 1]
        assert -1.0 <= calc.normalized <= 1.0

    def test_ofi_starts_zero(self):
        calc = OFICalculator(window=20)
        assert calc.normalized == 0.0

    def test_ofi_update_returns_cumulative(self):
        calc = OFICalculator(window=20)
        # First call initialises prev state, returns 0
        first = calc.update(2850.0, 2850.03, 100.0, 90.0)
        assert first == 0.0
        # Subsequent calls return cumulative OFI
        second = calc.update(2850.01, 2850.04, 110.0, 90.0)
        assert isinstance(second, float)
