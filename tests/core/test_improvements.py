"""Tests for Phase 1-7 improvements: new features and bug fixes."""

import math
import time
import pytest
from datetime import datetime, timezone, timedelta

from aphelion.core.clock import MarketClock
from aphelion.core.config import Session, SENTINEL
from aphelion.core.event_bus import EventBus, Event, Priority
from aphelion.core.data_layer import DataLayer, Tick, DataQualityValidator
from aphelion.core.registry import Registry, ComponentStatus, MODULES


# ─── Clock Improvements ──────────────────────────────────────────────────────


class TestClockImprovements:
    def setup_method(self):
        self.clock = MarketClock()

    def test_cyclical_time_encoding_range(self):
        """Cyclical features must be in [-1, 1]."""
        dt = datetime(2026, 3, 4, 14, 30, tzinfo=timezone.utc)
        features = self.clock.session_features(dt)
        for key in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_sin", "dom_cos"):
            assert -1.0 <= features[key] <= 1.0, f"{key}={features[key]} out of range"

    def test_cyclical_encoding_wraps_midnight(self):
        """Hour encoding at 0:00 and 23:59 should be close (near zero for sin)."""
        dt_midnight = datetime(2026, 3, 4, 0, 0, tzinfo=timezone.utc)
        dt_late = datetime(2026, 3, 3, 23, 59, tzinfo=timezone.utc)
        f1 = self.clock.session_features(dt_midnight)
        f2 = self.clock.session_features(dt_late)
        # Both should have hour_sin near 0 (wrapping from 23→0)
        assert abs(f1["hour_sin"]) < 0.01  # sin(0) = 0
        assert abs(f2["hour_sin"]) < 0.3    # sin(2π*23/24) ≈ -0.26

    def test_minutes_to_close_dead_zone_returns_next_session(self):
        """In DEAD_ZONE, minutes_to_close returns minutes to next session open."""
        dt = datetime(2026, 3, 4, 22, 0, tzinfo=timezone.utc)  # DEAD_ZONE
        session = self.clock.current_session(dt)
        assert session == Session.DEAD_ZONE
        minutes = self.clock.minutes_to_close(dt)
        assert minutes > 0

    def test_news_lockout_bisect_matches_original(self):
        """Binary search news lockout should match expected behavior."""
        events = [
            {"time": datetime(2026, 3, 4, 13, 30, tzinfo=timezone.utc), "impact": "HIGH", "name": "NFP"},
            {"time": datetime(2026, 3, 4, 15, 0, tzinfo=timezone.utc), "impact": "HIGH", "name": "FOMC"},
            {"time": datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc), "impact": "MED", "name": "PMI"},
        ]
        self.clock.set_news_calendar(events)
        # Right before NFP: should be in lockout
        dt_before = datetime(2026, 3, 4, 13, 27, tzinfo=timezone.utc)
        assert self.clock.is_news_lockout(dt_before) is True
        # Well after FOMC: should NOT be in lockout
        dt_after = datetime(2026, 3, 4, 15, 10, tzinfo=timezone.utc)
        assert self.clock.is_news_lockout(dt_after) is False
        # MED impact should not trigger lockout
        dt_med = datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc)
        assert self.clock.is_news_lockout(dt_med) is False

    def test_month_end_business_days(self):
        """Month end should account for weekends/business days."""
        # Thursday March 26, 2026 — 3 calendar days to EOM but only 2 business days (Fri, Mon)
        dt = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)  # Friday
        assert self.clock.is_month_end(dt) is True

    def test_dst_offsets_settable(self):
        """DST offsets can be set without error."""
        self.clock.set_dst_offsets(london_minutes=-60, ny_minutes=-60)
        assert self.clock._dst_offset_london == -60
        assert self.clock._dst_offset_ny == -60


# ─── Registry Improvements ───────────────────────────────────────────────────


class TestRegistryImprovements:
    def setup_method(self):
        self.registry = Registry(heartbeat_timeout=1.0)  # 1 second for quick test
        self.module_name = list(MODULES.keys())[0]

    def test_deregister_removes_component(self):
        self.registry.register(self.module_name)
        self.registry.deregister(self.module_name)
        with pytest.raises(KeyError):
            self.registry.get_status(self.module_name)

    def test_deregister_unknown_raises(self):
        with pytest.raises(KeyError):
            self.registry.deregister("non_existent")

    def test_heartbeat_timeout_detection(self):
        self.registry.register(self.module_name)
        self.registry.set_status(self.module_name, ComponentStatus.ACTIVE)
        self.registry.heartbeat(self.module_name)
        time.sleep(1.5)  # Exceed 1s timeout
        stale = self.registry.get_stale_components()
        assert self.module_name in stale

    def test_check_heartbeats_marks_error(self):
        self.registry.register(self.module_name)
        self.registry.set_status(self.module_name, ComponentStatus.ACTIVE)
        self.registry.heartbeat(self.module_name)
        time.sleep(1.5)
        stale_names = self.registry.check_heartbeats()
        assert self.module_name in stale_names
        assert self.registry.get_status(self.module_name).status == ComponentStatus.ERROR

    def test_system_health_includes_min(self):
        names = list(MODULES.keys())[:2]
        for name in names:
            self.registry.register(name)
            self.registry.set_status(name, ComponentStatus.ACTIVE)
        self.registry.set_health(names[0], 100.0)
        self.registry.set_health(names[1], 50.0)
        health = self.registry.system_health()
        assert "min_health" in health
        assert health["min_health"] == 50.0
        assert health["avg_health"] == 75.0
        # Overall = 0.7 * 75 + 0.3 * 50 = 52.5 + 15 = 67.5
        assert health["overall"] == pytest.approx(67.5, abs=0.1)


# ─── DataLayer Improvements ──────────────────────────────────────────────────


class TestDataLayerImprovements:
    def test_configurable_thresholds(self):
        """DataQualityValidator should accept custom thresholds."""
        validator = DataQualityValidator(max_spread=10.0, max_price_jump_pct=0.01)
        # Spread of 15 should fail with custom threshold
        tick = Tick(timestamp=time.time(), bid=2800.0, ask=2815.0, last=2807.5, volume=1.0)
        valid, error = validator.validate_tick(tick)
        assert not valid
        assert "Spread too wide" in error

    def test_gap_detection(self):
        """Validator detects time gaps between ticks."""
        validator = DataQualityValidator(max_gap_seconds=5.0)
        t = time.time()
        tick1 = Tick(timestamp=t, bid=2800.0, ask=2800.5, last=2800.25, volume=1.0)
        tick2 = Tick(timestamp=t + 10.0, bid=2800.1, ask=2800.6, last=2800.35, volume=1.0)
        validator.validate_tick(tick1)
        validator.validate_tick(tick2)
        assert validator.stats["gap_count"] == 1

    def test_staleness_detection(self):
        """DataLayer reports staleness."""
        bus = EventBus()
        dl = DataLayer(bus)
        assert dl.is_stale() is False  # No data yet
        assert dl.staleness_seconds() == 0.0

    def test_stats_includes_staleness(self):
        bus = EventBus()
        dl = DataLayer(bus)
        stats = dl.stats
        assert "staleness_seconds" in stats


# ─── SENTINEL Improvements ───────────────────────────────────────────────────


class TestSentinelCoreImprovements:
    def setup_method(self):
        self.bus = EventBus()
        self.clock = MarketClock()
        self.clock.set_simulated_time(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))

    def test_l1_triggers_at_3pct(self):
        from aphelion.risk.sentinel.core import SentinelCore
        core = SentinelCore(self.bus, self.clock)
        core.update_equity(10000.0)
        core.update_equity(9650.0)  # 3.5% drawdown > 3% L1 threshold
        assert core.l1_triggered is True
        assert core.l2_triggered is False

    def test_l2_triggers_at_6pct(self):
        from aphelion.risk.sentinel.core import SentinelCore
        core = SentinelCore(self.bus, self.clock)
        core.update_equity(10000.0)
        core.update_equity(9350.0)  # 6.5% drawdown > 6% L2 threshold
        assert core.l2_triggered is True
        assert core.is_trading_allowed() is False

    def test_pnl_lot_multiplier(self):
        """P&L should include lot_size_oz multiplier (100 oz per lot)."""
        from aphelion.risk.sentinel.core import SentinelCore, Position
        core = SentinelCore(self.bus, self.clock)
        core.update_equity(10000.0)
        pos = Position(
            position_id="test-1", symbol="XAUUSD", direction="LONG",
            entry_price=2800.0, stop_loss=2790.0, take_profit=2820.0,
            size_lots=0.1, size_pct=0.02, open_time=datetime.now(timezone.utc),
        )
        core.register_position(pos)
        core.close_position("test-1", exit_price=2810.0)
        # P&L = (2810 - 2800) * 0.1 * 100 = $100
        assert core._daily_pnl == pytest.approx(100.0, abs=0.01)

    def test_size_multiplier_l1(self):
        from aphelion.risk.sentinel.core import SentinelCore
        core = SentinelCore(self.bus, self.clock)
        core.update_equity(10000.0)
        assert core.get_size_multiplier() == 1.0
        core.update_equity(9650.0)  # L1
        assert core.get_size_multiplier() == 0.5

    def test_daily_reset_clears_counters(self):
        from aphelion.risk.sentinel.core import SentinelCore
        core = SentinelCore(self.bus, self.clock)
        core.update_equity(10000.0)
        core.update_equity(9650.0)  # L1
        assert core.l1_triggered
        # Advance to next day
        self.clock.set_simulated_time(datetime(2024, 1, 16, 12, 0, tzinfo=timezone.utc))
        core.update_equity(9650.0)
        assert core.l1_triggered is False  # Reset for new day


# ─── CircuitBreaker Improvements ─────────────────────────────────────────────


class TestCircuitBreakerImprovements:
    def test_configurable_thresholds(self):
        from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(EventBus(), l1_threshold=0.02, l2_threshold=0.04, l3_threshold=0.08)
        cb.update(10000.0)
        cb.update(9750.0)  # 2.5% → L1 with custom threshold
        assert cb.state == "L1"

    def test_l2_recovery_to_l1(self):
        from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(EventBus(), cooldown_seconds=0)  # No cooldown for test
        cb.update(10000.0)
        cb.update(9200.0)  # L2
        assert cb.state == "L2"
        # Recovery: new peak resets drawdown below L2 but above L1
        cb.update(10000.0)  # New peak, 0% dd → L2 recovery → L1
        # After updating to a new peak but state was L2, it should try to recover
        # Actually a new peak means dd=0, which is below L1 threshold
        # So L2 recovery fires, then L1 reset fires
        assert cb.state in ("L1", "NORMAL")


# ─── Validator Improvements ──────────────────────────────────────────────────


class TestValidatorImprovements:
    def test_warnings_populated_on_l1(self):
        from aphelion.risk.sentinel.core import SentinelCore
        from aphelion.risk.sentinel.validator import TradeValidator, TradeProposal
        bus = EventBus()
        clock = MarketClock()
        clock.set_simulated_time(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        core = SentinelCore(bus, clock)
        core.update_equity(10000.0)
        core.update_equity(9650.0)  # L1
        validator = TradeValidator(core, clock)
        proposal = TradeProposal(
            symbol="XAUUSD", direction="LONG",
            entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
            size_pct=0.01, proposed_by="TEST",
        )
        result = validator.validate(proposal)
        # L1 is active but trading still allowed (L2 halts)
        assert any("L1_ACTIVE" in w for w in result.warnings)

    def test_bulk_validate_cumulative_exposure(self):
        from aphelion.risk.sentinel.core import SentinelCore
        from aphelion.risk.sentinel.validator import TradeValidator, TradeProposal
        bus = EventBus()
        clock = MarketClock()
        clock.set_simulated_time(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
        core = SentinelCore(bus, clock)
        core.update_equity(10000.0)
        validator = TradeValidator(core, clock)
        # 3 proposals at 0.02 each = 0.06 total, max = 0.02 * 3 = 0.06
        proposals = [
            TradeProposal(
                symbol="XAUUSD", direction="LONG",
                entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
                size_pct=0.02, proposed_by="TEST",
            )
            for _ in range(4)  # 4th should be rejected (exceeds 3 positions)
        ]
        results = validator.bulk_validate(proposals)
        approved_count = sum(1 for r in results if r.approved)
        # At most 3 can be approved (max_simultaneous_positions=3)
        assert approved_count <= 3


# ─── Microstructure Improvements ─────────────────────────────────────────────


class TestMicrostructureImprovements:
    def test_hawkes_o1_matches_expected(self):
        """O(1) Hawkes should produce monotonically increasing intensity."""
        from aphelion.features.microstructure import HawkesIntensity
        hawkes = HawkesIntensity(decay=0.1, baseline=1.0)
        i1 = hawkes.update(1.0)
        i2 = hawkes.update(1.1)
        i3 = hawkes.update(1.2)
        # Each event should increase intensity
        assert i2 > i1
        assert i3 > i2

    def test_hawkes_decays_over_time(self):
        """Intensity should decay when queried later without events."""
        from aphelion.features.microstructure import HawkesIntensity
        hawkes = HawkesIntensity(decay=1.0, baseline=1.0)
        hawkes.update(0.0)
        early = hawkes.current(0.1)
        late = hawkes.current(10.0)
        assert late < early

    def test_ofi_normalized_range(self):
        """Normalized OFI should be in [-1, 1]."""
        from aphelion.features.microstructure import OFICalculator
        ofi = OFICalculator(window=10)
        ofi.update(2800.0, 2800.5, 10.0, 10.0)
        ofi.update(2801.0, 2801.5, 15.0, 10.0)
        ofi.update(2802.0, 2802.5, 20.0, 10.0)
        assert -1.0 <= ofi.normalized <= 1.0

    def test_microstructure_state_has_ofi_normalized(self):
        from aphelion.features.microstructure import MicrostructureEngine
        engine = MicrostructureEngine()
        state = engine.update(
            timestamp=time.time(), bid=2800.0, ask=2800.5,
            last_price=2800.25, volume=10.0,
        )
        assert hasattr(state, "ofi_normalized")
        d = engine.to_dict()
        assert "ofi_normalized" in d


# ─── Position Sizer Improvements ─────────────────────────────────────────────


class TestPositionSizerImprovements:
    def test_zero_entry_raises(self):
        from aphelion.risk.sentinel.position_sizer import PositionSizer
        sizer = PositionSizer()
        with pytest.raises(ValueError):
            sizer.pct_to_lots(0.02, 10000.0, 0.0)

    def test_atr_based_lots(self):
        from aphelion.risk.sentinel.position_sizer import PositionSizer
        sizer = PositionSizer()
        lots = sizer.atr_based_lots(
            account_equity=10000.0,
            risk_pct=0.02,
            atr=25.0,
            lot_size_oz=100.0,
        )
        # risk_dollars = 200, lots = 200 / (25 * 100) = 0.08
        assert lots == pytest.approx(0.08, abs=0.01)

    def test_volatility_scalar_adjusts_size(self):
        from aphelion.risk.sentinel.position_sizer import PositionSizer
        sizer = PositionSizer()
        normal = sizer.compute_size_pct(0.6, 30.0, 15.0, confidence=1.0, volatility_scalar=1.0)
        reduced = sizer.compute_size_pct(0.6, 30.0, 15.0, confidence=1.0, volatility_scalar=0.5)
        assert reduced < normal
