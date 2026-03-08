"""Shared fixtures for Phase 3 backtest tests."""

import numpy as np
import pytest
from datetime import datetime, timezone

from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar
from aphelion.core.event_bus import EventBus
from aphelion.core.clock import MarketClock
from aphelion.risk.sentinel.core import SentinelCore, Position
from aphelion.risk.sentinel.validator import TradeValidator, TradeProposal
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer
from aphelion.risk.sentinel.position_sizer import PositionSizer


def make_bar(ts_offset=0, open_=2850.0, high=2852.0, low=2848.0,
             close=2850.0, volume=100.0, tf=Timeframe.M1) -> Bar:
    return Bar(
        timestamp=datetime.fromtimestamp(1704067200.0 + ts_offset, tz=timezone.utc),
        timeframe=tf,
        open=open_, high=high, low=low, close=close,
        volume=volume, tick_volume=100, spread=0.20, is_complete=True,
    )


def make_bars(n=200, start_price=2850.0, seed=42) -> list:
    rng = np.random.default_rng(seed)
    bars = []
    price = start_price
    for i in range(n):
        price = max(100.0, price + float(rng.normal(0, 0.5)))
        spread = abs(float(rng.normal(0.3, 0.05)))
        high = price + abs(float(rng.normal(0, 0.3)))
        low = price - abs(float(rng.normal(0, 0.3)))
        high = max(high, price)
        low = min(low, price)
        if high <= low:
            high = low + 0.01
        bars.append(Bar(
            timestamp=datetime.fromtimestamp(1704067200.0 + i * 60, tz=timezone.utc),
            timeframe=Timeframe.M1,
            open=price - float(rng.normal(0, 0.1)),
            high=high, low=low, close=price,
            volume=float(rng.uniform(50, 200)),
            tick_volume=int(rng.integers(10, 100)),
            spread=spread, is_complete=True,
        ))
    return bars


def make_sentinel_stack():
    bus = EventBus()
    clock = MarketClock()
    # FIXED: Set simulated time to a known market-open period for deterministic tests
    clock.set_simulated_time(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc))
    core = SentinelCore(bus, clock)
    sizer = PositionSizer()
    validator = TradeValidator(core, clock)
    cb = CircuitBreaker(bus)
    enforcer = ExecutionEnforcer(validator, cb)
    return {
        "bus": bus, "clock": clock, "core": core,
        "sizer": sizer, "validator": validator,
        "cb": cb, "enforcer": enforcer,
    }


def make_valid_proposal(entry=2850.0, size_pct=0.02) -> TradeProposal:
    return TradeProposal(
        symbol="XAUUSD", direction="LONG",
        entry_price=entry, stop_loss=entry - 10.0,
        take_profit=entry + 20.0,
        size_pct=size_pct, proposed_by="TEST",
    )
