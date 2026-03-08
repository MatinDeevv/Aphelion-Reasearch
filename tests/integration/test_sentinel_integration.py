"""Integration tests: SENTINEL risk pipeline end-to-end."""

import pytest
from datetime import datetime, timezone

from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock
from aphelion.core.config import SENTINEL
from aphelion.risk.sentinel.core import SentinelCore, Position
from aphelion.risk.sentinel.validator import TradeProposal, TradeValidator
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer


def _make_stack():
    bus = EventBus()
    clock = MarketClock()
    clock.set_simulated_time(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc))
    core = SentinelCore(bus, clock)
    core.update_equity(10000.0)
    validator = TradeValidator(core, clock)
    cb = CircuitBreaker(bus)
    cb.update(10000.0)
    enforcer = ExecutionEnforcer(validator, cb)
    return {
        "bus": bus, "clock": clock, "core": core,
        "validator": validator, "cb": cb, "enforcer": enforcer,
    }


def _make_proposal(entry=2850.0, sl=2840.0, tp=2870.0, size_pct=0.02,
                    direction="LONG", symbol="XAUUSD"):
    return TradeProposal(
        symbol=symbol, direction=direction,
        entry_price=entry, stop_loss=sl, take_profit=tp,
        size_pct=size_pct, proposed_by="TEST",
    )


class TestSentinelIntegration:
    def test_valid_trade_gets_approved(self):
        stack = _make_stack()
        proposal = _make_proposal()
        result = stack["validator"].validate(proposal)
        assert result.approved
        assert len(result.rejections) == 0

    def test_trade_rejected_missing_stop_loss(self):
        stack = _make_stack()
        proposal = _make_proposal(sl=0.0)
        result = stack["validator"].validate(proposal)
        assert not result.approved
        assert any("NO_STOP_LOSS" in r for r in result.rejections)

    def test_trade_rejected_bad_rr(self):
        stack = _make_stack()
        # risk=10, reward=3 → RR=0.3 < 1.5
        proposal = _make_proposal(sl=2840.0, tp=2853.0)
        result = stack["validator"].validate(proposal)
        assert not result.approved
        assert any("INSUFFICIENT_RR" in r for r in result.rejections)

    def test_trade_rejected_at_max_positions(self):
        stack = _make_stack()
        core = stack["core"]
        for i in range(SENTINEL.max_simultaneous_positions):
            core.register_position(Position(
                position_id=f"pos-{i}", symbol="XAUUSD", direction="LONG",
                entry_price=2850.0, stop_loss=2840.0, take_profit=2870.0,
                size_lots=0.01, size_pct=0.02,
                open_time=datetime.now(timezone.utc),
            ))
        proposal = _make_proposal()
        result = stack["validator"].validate(proposal)
        assert not result.approved
        assert any("MAX_POSITIONS" in r for r in result.rejections)

    def test_l3_blocks_all_trades(self):
        stack = _make_stack()
        core = stack["core"]
        core.update_equity(10000.0)
        core.update_equity(8900.0)  # >10% drawdown → L3
        assert core.l3_triggered
        proposal = _make_proposal()
        result = stack["validator"].validate(proposal)
        assert not result.approved
        assert any("TRADING_HALTED" in r for r in result.rejections)

    def test_circuit_breaker_reduces_size_in_l2(self):
        stack = _make_stack()
        cb = stack["cb"]
        cb.update(10000.0)
        cb.update(9200.0)  # 8% drawdown → L2
        assert cb.state == "L2"
        assert cb.size_multiplier == 0.25
        result = cb.apply_multiplier(0.02)
        assert result == pytest.approx(0.005, abs=0.001)

    def test_sl_breach_publishes_critical_event(self):
        stack = _make_stack()
        core = stack["core"]
        core.update_equity(10000.0)
        core.update_equity(8900.0)  # triggers L3
        assert core.l3_triggered

    def test_full_pipeline_valid_trade_then_rejection(self):
        stack = _make_stack()
        enforcer = stack["enforcer"]
        # First: valid trade
        approved, _, size = enforcer.approve_order(_make_proposal())
        assert approved
        assert size > 0
        # Second: bad RR trade
        bad = _make_proposal(sl=2840.0, tp=2853.0)
        approved2, reason, _ = enforcer.approve_order(bad)
        assert not approved2

    def test_rejection_rate_tracked(self):
        stack = _make_stack()
        enforcer = stack["enforcer"]
        enforcer.approve_order(_make_proposal())
        enforcer.approve_order(_make_proposal(sl=0.0))
        stats = enforcer.get_rejection_summary()
        assert stats["approved_count"] == 1
        assert stats["rejected_count"] == 1
        assert stats["rejection_rate"] == pytest.approx(0.5, abs=0.01)

    def test_all_sentinel_rules_run_on_invalid_proposal(self):
        stack = _make_stack()
        proposal = _make_proposal(sl=0.0, symbol="EURUSD")
        result = stack["validator"].validate(proposal)
        assert not result.approved
        assert len(result.rejections) >= 2

    def test_trade_rejected_oversized(self):
        stack = _make_stack()
        proposal = _make_proposal(size_pct=0.05)  # > SENTINEL 2%
        result = stack["validator"].validate(proposal)
        assert not result.approved
        assert any("SIZE_EXCEEDED" in r for r in result.rejections)

    def test_rejection_metrics_reset(self):
        stack = _make_stack()
        enforcer = stack["enforcer"]
        enforcer.approve_order(_make_proposal(sl=0.0))
        assert enforcer.get_rejection_summary()["rejected_count"] == 1

    def test_l1_reduces_size_by_half(self):
        stack = _make_stack()
        cb = stack["cb"]
        cb.update(10000.0)
        cb.update(9450.0)  # 5.5% drawdown → L1
        assert cb.state == "L1"
        assert cb.size_multiplier == 0.50
        result = cb.apply_multiplier(0.02)
        assert result == pytest.approx(0.01, abs=0.001)

    def test_valid_trade_approved_with_correct_size(self):
        stack = _make_stack()
        enforcer = stack["enforcer"]
        approved, _, final_size = enforcer.approve_order(_make_proposal())
        assert approved
        assert final_size == pytest.approx(0.02, abs=0.001)
