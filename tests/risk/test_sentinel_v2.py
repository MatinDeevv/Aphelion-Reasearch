"""Tests for SENTINEL v2 — CorrelationGuard, LatencyMonitor, CascadeProtection."""

import pytest
import numpy as np

from aphelion.risk.sentinel.sentinel_v2 import (
    SentinelV2, CorrelationGuard, LatencyMonitor,
    CascadeProtection, DynamicSizer,
)


# ── CorrelationGuard ──────────────────────────────────────────────


class TestCorrelationGuard:
    def test_can_open_first(self):
        guard = CorrelationGuard()
        ok, msg = guard.can_open("LONG")
        assert ok is True

    def test_max_same_direction(self):
        guard = CorrelationGuard()
        guard.register_position("p1", "LONG")
        guard.register_position("p2", "LONG")
        ok, msg = guard.can_open("LONG")
        assert ok is False
        assert "MAX_SAME_DIRECTION" in msg

    def test_max_total(self):
        guard = CorrelationGuard()
        guard.register_position("p1", "LONG")
        guard.register_position("p2", "SHORT")
        guard.register_position("p3", "LONG")
        ok, msg = guard.can_open("SHORT")
        assert ok is False
        assert "MAX_POSITIONS" in msg

    def test_remove_position(self):
        guard = CorrelationGuard()
        guard.register_position("p1", "LONG")
        guard.register_position("p2", "LONG")
        guard.remove_position("p1")
        ok, msg = guard.can_open("LONG")
        assert ok is True


# ── LatencyMonitor ────────────────────────────────────────────────


class TestLatencyMonitor:
    def test_no_halt_initially(self):
        monitor = LatencyMonitor()
        assert monitor.is_halted is False

    def test_normal_latency(self):
        monitor = LatencyMonitor()
        for _ in range(20):
            monitor.record(50.0)
        assert monitor.is_halted is False

    def test_high_latency_halt(self):
        monitor = LatencyMonitor()
        for _ in range(20):
            monitor.record(300.0)
        assert monitor.is_halted is True

    def test_p99_calculation(self):
        monitor = LatencyMonitor()
        for i in range(100):
            monitor.record(float(i))
        assert monitor.p99 > 0


# ── CascadeProtection ────────────────────────────────────────────


class TestCascadeProtection:
    def test_no_cascade_single(self):
        cp = CascadeProtection()
        result = cp.report_failure("MODULE_A")
        assert result is False

    def test_cascade_multiple(self):
        cp = CascadeProtection()
        cp.report_failure("MODULE_A")
        cp.report_failure("MODULE_B")
        result = cp.report_failure("MODULE_C")
        assert result is True
        assert cp.cascade_active is True

    def test_reset(self):
        cp = CascadeProtection()
        cp.report_failure("A")
        cp.report_failure("B")
        cp.report_failure("C")
        cp.reset()
        assert cp.cascade_active is False


# ── DynamicSizer ──────────────────────────────────────────────────


class TestDynamicSizer:
    def test_normal_regime(self):
        sizer = DynamicSizer()
        adjusted = sizer.compute_adjusted_size(0.02, "RANGING", 10.0, 10.0)
        assert 0 < adjusted <= 0.05

    def test_crisis_zero(self):
        sizer = DynamicSizer()
        adjusted = sizer.compute_adjusted_size(0.02, "CRISIS", 10.0, 10.0)
        assert adjusted == 0.0

    def test_high_vol_reduces(self):
        sizer = DynamicSizer()
        normal = sizer.compute_adjusted_size(0.02, "RANGING", 10.0, 10.0)
        high_vol = sizer.compute_adjusted_size(0.02, "RANGING", 20.0, 10.0)
        assert high_vol < normal

    def test_trending_increases(self):
        sizer = DynamicSizer()
        ranging = sizer.compute_adjusted_size(0.02, "RANGING", 10.0, 10.0)
        trending = sizer.compute_adjusted_size(0.02, "TRENDING_BULL", 10.0, 10.0)
        assert trending > ranging


# ── SentinelV2 Integration ────────────────────────────────────────


class TestSentinelV2:
    def test_trade_allowed_clean(self):
        sv2 = SentinelV2()
        ok, msg = sv2.is_trade_allowed("LONG")
        assert ok is True

    def test_blocked_cascade(self):
        sv2 = SentinelV2()
        sv2.cascade_protection.report_failure("A")
        sv2.cascade_protection.report_failure("B")
        sv2.cascade_protection.report_failure("C")
        ok, msg = sv2.is_trade_allowed("LONG")
        assert ok is False
        assert "CASCADE" in msg

    def test_blocked_latency(self):
        sv2 = SentinelV2()
        for _ in range(20):
            sv2.latency_monitor.record(500.0)
        ok, msg = sv2.is_trade_allowed("LONG")
        assert ok is False
        assert "LATENCY" in msg

    def test_blocked_correlation(self):
        sv2 = SentinelV2()
        sv2.correlation_guard.register_position("p1", "LONG")
        sv2.correlation_guard.register_position("p2", "LONG")
        ok, msg = sv2.is_trade_allowed("LONG")
        assert ok is False
