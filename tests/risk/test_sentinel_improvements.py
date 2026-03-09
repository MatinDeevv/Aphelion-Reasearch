"""Tests for risk/sentinel improvements: Kelly formula, Friday close, exports."""

import pytest
import math
from aphelion.risk.sentinel.position_sizer import PositionSizer


class TestKellyFormula:
    def test_kelly_scale_invariant(self):
        """Kelly fraction should be the same regardless of absolute price levels."""
        sizer = PositionSizer()
        # 60% win rate, avg win $200 / avg loss $100 (R:R = 2:1)
        k1 = sizer.kelly_fraction(win_rate=0.60, avg_win=200.0, avg_loss=100.0)
        # Same ratio at 10x scale
        k2 = sizer.kelly_fraction(win_rate=0.60, avg_win=2000.0, avg_loss=1000.0)
        assert k1 == pytest.approx(k2, rel=0.01)

    def test_kelly_zero_win_rate(self):
        k = PositionSizer().kelly_fraction(win_rate=0.0, avg_win=100.0, avg_loss=50.0)
        assert k <= 0.0

    def test_kelly_edge_cases(self):
        sizer = PositionSizer()
        # 100% win rate — full Kelly = 1.0, quarter-Kelly = 0.25, capped at KELLY_MAX_F = 0.02
        k = sizer.kelly_fraction(win_rate=1.0, avg_win=100.0, avg_loss=50.0)
        assert k == pytest.approx(0.02, rel=0.01)

    def test_kelly_50_50_equal_payoff(self):
        """50/50 with equal win/loss -> Kelly = 0 (no edge)."""
        k = PositionSizer().kelly_fraction(win_rate=0.50, avg_win=100.0, avg_loss=100.0)
        assert abs(k) < 0.01

    def test_kelly_positive_expectancy(self):
        """Positive expectancy should yield positive Kelly."""
        k = PositionSizer().kelly_fraction(win_rate=0.55, avg_win=150.0, avg_loss=100.0)
        assert k > 0.0


class TestSentinelExports:
    def test_circuit_breaker_importable(self):
        from aphelion.risk.sentinel import CircuitBreaker
        assert CircuitBreaker is not None

    def test_execution_enforcer_importable(self):
        from aphelion.risk.sentinel import ExecutionEnforcer
        assert ExecutionEnforcer is not None

    def test_all_exports(self):
        import aphelion.risk.sentinel as sentinel
        assert hasattr(sentinel, "SentinelCore")
        assert hasattr(sentinel, "TradeValidator")
        assert hasattr(sentinel, "CircuitBreaker")
        assert hasattr(sentinel, "ExecutionEnforcer")
        assert hasattr(sentinel, "PositionSizer")
