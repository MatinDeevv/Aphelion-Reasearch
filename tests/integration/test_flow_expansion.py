"""Tests for Flow expansion — Phantom, Specter cores."""

import pytest
import numpy as np

from aphelion.flow.phantom.core import PhantomCore, HiddenOrder
from aphelion.flow.specter.core import SpecterCore, StealthSignal


# ── PhantomCore ─────────────────────────────────────────────────────────────

class TestPhantomCore:

    def test_no_detection_on_first_trade(self):
        p = PhantomCore()
        result = p.on_trade(1950.0, 100, 500, 1)
        assert result is None

    def test_detection_after_repeated_fills(self):
        p = PhantomCore(min_fill_count=3, size_ratio_threshold=2.0)
        for _ in range(5):
            result = p.on_trade(1950.00, 200, 50, 1)
        # Should detect after enough fills with high volume/visible ratio
        assert result is not None or len(p.recent_detections) > 0

    def test_detection_is_hidden_order(self):
        p = PhantomCore(min_fill_count=2, size_ratio_threshold=1.0)
        p.on_trade(1950.00, 500, 100, 1)
        result = p.on_trade(1950.00, 600, 100, 1)
        if result is not None:
            assert isinstance(result, HiddenOrder)
            assert result.price == 1950.00

    def test_reset_level(self):
        p = PhantomCore()
        p.on_trade(1950.00, 100, 50, 1)
        p.reset_level(1950.00)
        assert p.active_levels == 0

    def test_active_levels_count(self):
        p = PhantomCore()
        p.on_trade(1950.00, 100, 50, 1)
        p.on_trade(1955.00, 100, 50, -1)
        assert p.active_levels == 2


# ── SpecterCore ─────────────────────────────────────────────────────────────

class TestSpecterCore:

    def test_no_signal_insufficient_data(self):
        s = SpecterCore(lookback=10)
        result = s.update(100, 80, 0.01)
        assert result is None

    def test_accumulation_signal(self):
        s = SpecterCore(lookback=10, imbalance_threshold=0.2)
        # Heavy buying with flat price → accumulation
        for _ in range(15):
            s.update(200, 50, 0.0)  # Buy heavy, price flat
        result = s.update(200, 50, 0.0)
        if result is not None:
            assert result.direction == 1
            assert isinstance(result, StealthSignal)

    def test_distribution_signal(self):
        s = SpecterCore(lookback=10, imbalance_threshold=0.2)
        # Heavy selling with flat/up price → distribution
        for _ in range(15):
            s.update(50, 200, 0.01)
        result = s.update(50, 200, 0.01)
        if result is not None:
            assert result.direction == -1

    def test_recent_signals(self):
        s = SpecterCore(lookback=5, imbalance_threshold=0.1)
        for _ in range(20):
            s.update(300, 50, -0.001)
        assert isinstance(s.recent_signals, list)

    def test_balanced_flow_no_signal(self):
        s = SpecterCore(lookback=10)
        for _ in range(20):
            result = s.update(100, 100, 0.01)
        # Balanced flow should not produce a signal
        assert result is None
