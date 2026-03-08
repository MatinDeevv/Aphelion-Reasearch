"""Tests for aphelion.risk.sentinel.circuit_breaker — CircuitBreaker."""

import pytest
from aphelion.core.event_bus import EventBus
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_initial_state_normal(self):
        cb = CircuitBreaker(EventBus())
        assert cb.state == "NORMAL"
        assert cb.size_multiplier == 1.0

    def test_l1_triggers_at_5pct(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(9450.0)  # 5.5% drawdown
        assert cb.state == "L1"
        assert cb.size_multiplier == 0.50

    def test_l2_triggers_at_7_5pct(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(9200.0)  # 8% drawdown
        assert cb.state == "L2"
        assert cb.size_multiplier == 0.25

    def test_l3_triggers_at_10pct(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(8900.0)  # 11% drawdown
        assert cb.state == "L3"
        assert cb.size_multiplier == 0.0

    def test_no_double_trigger_l3(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(8900.0)  # L3
        cb.update(8800.0)  # Still L3, no re-trigger
        assert cb.state == "L3"
        # Only one L3 trigger entry
        l3_triggers = [t for t in cb.get_summary()["trigger_history"] if t["level"] == "L3"]
        assert len(l3_triggers) == 1

    def test_reset_from_l1(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(9450.0)  # L1
        assert cb.state == "L1"
        cb.update(10100.0)  # Recovery - new peak, 0% dd
        assert cb.state == "NORMAL"

    def test_cannot_reset_from_l3(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(8900.0)  # L3
        cb.reset()  # Should be no-op (only resets from L1)
        assert cb.state == "L3"

    def test_apply_multiplier_scales_size(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(9450.0)  # L1 → 50%
        result = cb.apply_multiplier(0.02)
        assert result == pytest.approx(0.01, abs=0.001)

    def test_apply_multiplier_l3_returns_zero(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(8900.0)  # L3 → 0%
        result = cb.apply_multiplier(0.02)
        assert result == 0.0

    def test_summary_dict_keys(self):
        cb = CircuitBreaker(EventBus())
        s = cb.get_summary()
        assert "state" in s
        assert "size_multiplier" in s
        assert "current_drawdown" in s

    def test_peak_never_decreases(self):
        cb = CircuitBreaker(EventBus())
        cb.update(10000.0)
        cb.update(9500.0)
        cb.update(9800.0)
        assert cb.get_summary()["peak_equity"] == 10000.0
