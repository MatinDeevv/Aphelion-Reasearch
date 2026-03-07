"""Tests for APHELION Microstructure Features."""

import numpy as np
from aphelion.features.microstructure import (
    VPINCalculator, OFICalculator, TickEntropyCalculator,
    HawkesIntensity, MicrostructureEngine,
)


class TestVPIN:
    def test_initial_zero(self):
        vpin = VPINCalculator(bucket_size=10)
        result = vpin.update(2850.0, 5.0)
        assert result == 0.0

    def test_increasing_on_directional(self):
        vpin = VPINCalculator(bucket_size=10)
        values = []
        # All upticks = high VPIN
        for i in range(100):
            v = vpin.update(2850.0 + i * 0.01, 5.0)
            values.append(v)
        # VPIN should be elevated with pure directional flow
        assert values[-1] > 0.5

    def test_low_on_balanced(self):
        vpin = VPINCalculator(bucket_size=10)
        values = []
        # Alternating up/down = low VPIN
        for i in range(100):
            price = 2850.0 + (0.01 if i % 2 == 0 else -0.01)
            v = vpin.update(price, 5.0)
            values.append(v)
        assert values[-1] < 0.5


class TestOFI:
    def test_initial_zero(self):
        ofi = OFICalculator()
        result = ofi.update(2850.0, 2850.50)
        assert result == 0.0

    def test_positive_on_bid_increase(self):
        ofi = OFICalculator()
        ofi.update(2850.0, 2850.50)
        result = ofi.update(2850.10, 2850.50)  # Bid up, ask same
        assert result > 0


class TestTickEntropy:
    def test_initial_max(self):
        entropy = TickEntropyCalculator()
        result = entropy.update(2850.0)
        assert result == 1.0

    def test_low_on_directional(self):
        entropy = TickEntropyCalculator(window=50)
        for i in range(60):
            entropy.update(2850.0 + i * 0.01)  # All up
        result = entropy.update(2850.0 + 60 * 0.01)
        assert result < 0.2  # Very directional

    def test_high_on_noise(self):
        entropy = TickEntropyCalculator(window=50)
        np.random.seed(42)
        for i in range(60):
            entropy.update(2850.0 + np.random.choice([-0.01, 0.01]))
        result = entropy.update(2850.0)
        assert result > 0.7  # Noisy


class TestHawkes:
    def test_baseline_intensity(self):
        hawkes = HawkesIntensity(baseline=1.0)
        assert hawkes.intensity == 1.0

    def test_increases_on_events(self):
        hawkes = HawkesIntensity(decay=0.1, baseline=1.0)
        t1 = hawkes.update(1.0)
        t2 = hawkes.update(1.5)
        assert t2 > t1  # More events = higher intensity


class TestMicrostructureEngine:
    def test_full_update(self):
        engine = MicrostructureEngine()
        state = engine.update(
            timestamp=1000.0,
            bid=2850.0, ask=2850.50,
            last_price=2850.25, volume=10.0,
        )
        assert state.bid_ask_spread == 0.5

    def test_to_dict(self):
        engine = MicrostructureEngine()
        engine.update(1000.0, 2850.0, 2850.50, 2850.25, 10.0)
        d = engine.to_dict()
        assert "vpin" in d
        assert "ofi" in d
        assert "tick_entropy" in d
        assert "hawkes_buy_intensity" in d
        assert "bid_ask_spread" in d
        assert "quote_depth" in d

    def test_quote_depth(self):
        engine = MicrostructureEngine()
        state = engine.update(
            timestamp=1000.0,
            bid=2850.0, ask=2850.50,
            last_price=2850.25, volume=10.0,
            bid_size=150.0, ask_size=200.0,
        )
        assert state.quote_depth == 350.0

    def test_quote_depth_default(self):
        engine = MicrostructureEngine()
        # Default bid_size=1.0, ask_size=1.0
        state = engine.update(1000.0, 2850.0, 2850.50, 2850.25, 10.0)
        assert state.quote_depth == 2.0
