"""Integration tests for tick -> bar -> feature pipeline."""

import asyncio

import numpy as np
import pytest

from aphelion.core.config import EventTopic, SENTINEL, Timeframe
from aphelion.core.data_layer import Bar, DataLayer, Tick
from aphelion.core.event_bus import Event, EventBus, Priority
from aphelion.features.engine import FeatureEngine


def build_ticks(n_ticks: int = 200) -> list[Tick]:
    """Create deterministic synthetic XAUUSD ticks."""
    rng = np.random.default_rng(7)
    ticks: list[Tick] = []
    price = 2850.0
    start_ts = 1704067200.0  # minute-aligned

    for i in range(n_ticks):
        price += float(rng.normal(0.0, 0.02))
        ticks.append(
            Tick(
                timestamp=start_ts + i,
                bid=price - 0.10,
                ask=price + 0.10,
                last=price,
                volume=1.0 + float(rng.uniform(0.0, 0.5)),
            )
        )
    return ticks


@pytest.mark.asyncio
async def test_tick_triggers_bar_event():
    bus = EventBus()
    data_layer = DataLayer(bus)
    received: list[Bar] = []

    async def on_bar(event: Event) -> None:
        received.append(event.data)

    bus.subscribe(EventTopic.BAR, on_bar)
    await bus.start()

    for tick in build_ticks(130):
        await data_layer.process_tick(tick)

    await asyncio.sleep(0.1)
    await bus.stop()

    assert received
    assert isinstance(received[0], Bar)


@pytest.mark.asyncio
async def test_tick_triggers_tick_event():
    bus = EventBus()
    data_layer = DataLayer(bus)
    received: list[Tick] = []

    async def on_tick(event: Event) -> None:
        received.append(event.data)

    bus.subscribe(EventTopic.TICK, on_tick)
    await bus.start()

    tick = build_ticks(1)[0]
    await data_layer.process_tick(tick)

    await asyncio.sleep(0.05)
    await bus.stop()

    assert len(received) == 1
    assert isinstance(received[0], Tick)


@pytest.mark.asyncio
async def test_invalid_tick_no_event():
    bus = EventBus()
    data_layer = DataLayer(bus)
    received: list[Tick] = []

    async def on_tick(event: Event) -> None:
        received.append(event.data)

    bus.subscribe(EventTopic.TICK, on_tick)
    await bus.start()

    invalid_tick = Tick(
        timestamp=1704067200.0,
        bid=-1.0,
        ask=2850.1,
        last=2850.0,
        volume=1.0,
    )
    await data_layer.process_tick(invalid_tick)

    await asyncio.sleep(0.05)
    await bus.stop()

    assert received == []


@pytest.mark.asyncio
async def test_multiple_timeframes():
    bus = EventBus()
    data_layer = DataLayer(bus)

    for tick in build_ticks(400):
        await data_layer.process_tick(tick)

    m1_bars = data_layer.get_bars(Timeframe.M1, count=100)
    m5_bars = data_layer.get_bars(Timeframe.M5, count=100)

    assert len(m1_bars) >= 5
    assert len(m5_bars) >= 1


@pytest.mark.asyncio
async def test_feature_engine_wired_to_data_layer():
    bus = EventBus()
    data_layer = DataLayer(bus)
    fe = FeatureEngine(data_layer)
    emitted_features: list[dict] = []

    async def on_bar(event: Event) -> None:
        emitted_features.append(fe.on_bar(event.data))

    bus.subscribe(EventTopic.BAR, on_bar)
    await bus.start()

    for tick in build_ticks(200):
        await data_layer.process_tick(tick)

    await asyncio.sleep(0.1)
    await bus.stop()

    assert emitted_features
    assert "vpin" in emitted_features[-1]


@pytest.mark.asyncio
async def test_event_bus_priority():
    bus = EventBus()
    processed_order: list[Priority] = []

    async def on_system(event: Event) -> None:
        processed_order.append(event.priority)

    bus.subscribe(EventTopic.SYSTEM, on_system)

    await bus.publish(
        Event(
            topic=EventTopic.SYSTEM,
            data="low",
            source="TEST",
            priority=Priority.LOW,
        )
    )
    await bus.publish(
        Event(
            topic=EventTopic.SYSTEM,
            data="critical",
            source="TEST",
            priority=Priority.CRITICAL,
        )
    )

    await bus.start()
    await asyncio.sleep(0.1)
    await bus.stop()

    assert processed_order[:2] == [Priority.CRITICAL, Priority.LOW]


def test_sentinel_limits_immutable():
    with pytest.raises((AttributeError, TypeError)):
        SENTINEL.max_position_pct = 0.03  # type: ignore[misc]


@pytest.mark.asyncio
async def test_full_stack_no_exception():
    bus = EventBus()
    data_layer = DataLayer(bus)
    fe = FeatureEngine(data_layer)

    async def on_bar(event: Event) -> None:
        fe.on_bar(event.data)

    bus.subscribe(EventTopic.BAR, on_bar)
    await bus.start()

    for tick in build_ticks(500):
        await data_layer.process_tick(tick)

    await asyncio.sleep(0.2)
    await bus.stop()

    assert data_layer.get_bars(Timeframe.M1, count=1000)
    assert bus.stats["errors"] == 0


@pytest.mark.asyncio
async def test_bar_event_payload_timeframe_is_m1():
    bus = EventBus()
    data_layer = DataLayer(bus)
    received: list[Bar] = []

    async def on_bar(event: Event) -> None:
        received.append(event.data)

    bus.subscribe(EventTopic.BAR, on_bar)
    await bus.start()

    for tick in build_ticks(130):
        await data_layer.process_tick(tick)

    await asyncio.sleep(0.1)
    await bus.stop()

    assert received
    assert received[0].timeframe == Timeframe.M1
