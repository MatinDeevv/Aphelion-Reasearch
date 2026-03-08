"""Tests for aphelion.risk.sentinel.core — SentinelCore."""

import asyncio
from datetime import datetime, timezone

import pytest

from aphelion.core.clock import MarketClock
from aphelion.core.config import SENTINEL
from aphelion.core.event_bus import EventBus
from aphelion.risk.sentinel.core import Position, SentinelCore


@pytest.fixture
def make_core():
    bus = EventBus()
    clock = MarketClock()
    # Set simulated time to a known market-open moment
    clock.set_simulated_time(datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc))
    core = SentinelCore(event_bus=bus, clock=clock)
    return core, bus, clock


def _make_position(pid: str = "P1", size_pct: float = 0.01) -> Position:
    return Position(
        position_id=pid,
        symbol="XAUUSD",
        direction="LONG",
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        size_lots=0.10,
        size_pct=size_pct,
        open_time=datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),
    )


class TestSentinelCore:
    def test_initial_state(self, make_core):
        core, bus, clock = make_core
        assert core.get_open_position_count() == 0
        assert core.get_total_exposure_pct() == 0.0
        assert core.l3_triggered is False

    def test_update_equity_sets_peak(self, make_core):
        core, bus, clock = make_core
        core.update_equity(10000.0)
        core.update_equity(10500.0)
        status = core.get_status()
        assert status["session_peak_equity"] == 10500.0

    def test_l3_triggers_on_10pct_drawdown(self, make_core):
        core, bus, clock = make_core
        core.update_equity(10000.0)
        core.update_equity(8900.0)  # > 10% drawdown
        assert core.l3_triggered is True

    def test_l3_not_triggered_on_small_drop(self, make_core):
        core, bus, clock = make_core
        core.update_equity(10000.0)
        core.update_equity(9500.0)  # 5% drawdown < 10%
        assert core.l3_triggered is False

    @pytest.mark.asyncio
    async def test_l3_publishes_critical_event(self, make_core):
        core, bus, clock = make_core
        received = []

        async def listener(event):
            received.append(event)

        from aphelion.core.config import EventTopic
        bus.subscribe(EventTopic.RISK, listener)

        await bus.start()
        try:
            core.update_equity(10000.0)
            core.update_equity(8900.0)
            # Allow dispatch loop to process queued events
            await asyncio.sleep(0.1)
        finally:
            await bus.stop()

        # At least one L3 event should have been published
        l3_events = [e for e in received if e.data.get("action") == "L3_DISCONNECT"]
        assert len(l3_events) >= 1

    def test_register_and_count_positions(self, make_core):
        core, bus, clock = make_core
        core.register_position(_make_position("P1"))
        core.register_position(_make_position("P2"))
        assert core.get_open_position_count() == 2

    def test_close_position_removes_it(self, make_core):
        core, bus, clock = make_core
        core.register_position(_make_position("P1"))
        assert core.get_open_position_count() == 1
        core.close_position("P1", exit_price=2010.0)
        assert core.get_open_position_count() == 0

    def test_total_exposure_sums_positions(self, make_core):
        core, bus, clock = make_core
        core.register_position(_make_position("P1", size_pct=0.01))
        core.register_position(_make_position("P2", size_pct=0.02))
        assert core.get_total_exposure_pct() == pytest.approx(0.03)

    def test_trading_halted_during_l3(self, make_core):
        core, bus, clock = make_core
        assert core.is_trading_allowed() is True
        core.update_equity(10000.0)
        core.update_equity(8900.0)
        assert core.is_trading_allowed() is False

    def test_status_dict_has_all_keys(self, make_core):
        core, bus, clock = make_core
        core.update_equity(10000.0)
        status = core.get_status()
        expected_keys = {
            "l1_triggered", "l2_triggered", "l3_triggered",
            "open_positions", "total_exposure_pct", "daily_pnl",
            "account_equity", "session_peak_equity",
            "trading_allowed", "current_drawdown_pct",
        }
        assert expected_keys.issubset(status.keys())
