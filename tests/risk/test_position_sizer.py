"""Tests for aphelion.risk.sentinel.position_sizer — PositionSizer."""

import pytest
from aphelion.core.config import KELLY_FRACTION, KELLY_MAX_F, SENTINEL
from aphelion.risk.sentinel.position_sizer import PositionSizer


class TestPositionSizer:
    def test_kelly_hard_cap(self):
        sizer = PositionSizer()
        # Extreme win rate → full kelly would be huge, must be capped at KELLY_MAX_F
        k = sizer.kelly_fraction(win_rate=0.99, avg_win=100.0, avg_loss=1.0)
        assert k <= KELLY_MAX_F

    def test_kelly_quarter_multiplier(self):
        sizer = PositionSizer()
        k = sizer.kelly_fraction(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
        assert k > 0
        assert k <= KELLY_MAX_F  # 0.02 hard cap

    def test_kelly_zero_on_negative(self):
        sizer = PositionSizer()
        # Negative expectancy → 0
        k = sizer.kelly_fraction(win_rate=0.30, avg_win=1.0, avg_loss=2.0)
        assert k == 0.0

    def test_kelly_handles_zero_avg_win(self):
        sizer = PositionSizer()
        # avg_win=0 → early return 0
        k = sizer.kelly_fraction(win_rate=0.50, avg_win=0.0, avg_loss=1.0)
        assert k == 0.0

    def test_compute_size_pct_capped(self):
        sizer = PositionSizer()
        pct = sizer.compute_size_pct(
            win_rate=0.99, avg_win=100.0, avg_loss=1.0, confidence=1.0
        )
        assert pct <= SENTINEL.max_position_pct  # 0.02

    def test_compute_size_pct_with_confidence(self):
        sizer = PositionSizer()
        pct_high = sizer.compute_size_pct(
            win_rate=0.60, avg_win=2.0, avg_loss=1.0, confidence=1.0
        )
        pct_low = sizer.compute_size_pct(
            win_rate=0.60, avg_win=2.0, avg_loss=1.0, confidence=0.5
        )
        assert pct_high >= pct_low

    def test_pct_to_lots_basic(self):
        sizer = PositionSizer()
        lots = sizer.pct_to_lots(
            size_pct=0.02, account_equity=10000.0, entry_price=2000.0
        )
        assert lots >= 0.01

    def test_pct_to_lots_minimum(self):
        sizer = PositionSizer()
        # Tiny risk → should clamp to min 0.01
        lots = sizer.pct_to_lots(
            size_pct=0.0001, account_equity=1000.0, entry_price=2000.0
        )
        assert lots == 0.01

    def test_validate_size_ok(self):
        sizer = PositionSizer()
        ok, reason = sizer.validate_size(0.01, current_exposure_pct=0.0)
        assert ok is True
        assert reason == "OK"

    def test_validate_size_exceeds_max(self):
        sizer = PositionSizer()
        ok, reason = sizer.validate_size(0.05, current_exposure_pct=0.0)
        assert ok is False
        assert "SIZE_EXCEEDED" in reason

    def test_validate_total_exposure_exceeded(self):
        sizer = PositionSizer()
        ok, reason = sizer.validate_size(0.02, current_exposure_pct=0.05)
        assert ok is False
        assert "EXPOSURE_EXCEEDED" in reason
