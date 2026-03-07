"""Tests for APHELION Event Bus."""

import asyncio
import pytest
from aphelion.core.event_bus import EventBus, Event, Priority
from aphelion.core.config import EventTopic


class TestEventBus:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventTopic.TICK, handler)
        event = Event(topic=EventTopic.TICK, data={"price": 2850.0}, source="TEST")

        await bus.publish(event)
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 1
        assert received[0].data["price"] == 2850.0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus):
        count = [0]

        async def handler1(event: Event):
            count[0] += 1

        async def handler2(event: Event):
            count[0] += 1

        bus.subscribe(EventTopic.BAR, handler1)
        bus.subscribe(EventTopic.BAR, handler2)

        await bus.publish(Event(topic=EventTopic.BAR, data={}, source="TEST"))
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        assert count[0] == 2

    @pytest.mark.asyncio
    async def test_topic_isolation(self, bus):
        tick_received = []
        bar_received = []

        async def tick_handler(event: Event):
            tick_received.append(event)

        async def bar_handler(event: Event):
            bar_received.append(event)

        bus.subscribe(EventTopic.TICK, tick_handler)
        bus.subscribe(EventTopic.BAR, bar_handler)

        await bus.publish(Event(topic=EventTopic.TICK, data={}, source="TEST"))
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(tick_received) == 1
        assert len(bar_received) == 0

    def test_priority_ordering(self):
        critical = Event(topic=EventTopic.RISK, data={}, source="SENTINEL", priority=Priority.CRITICAL)
        normal = Event(topic=EventTopic.TICK, data={}, source="DATA", priority=Priority.NORMAL)
        assert critical < normal

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventTopic.TICK, handler)
        bus.unsubscribe(EventTopic.TICK, handler)

        await bus.publish(Event(topic=EventTopic.TICK, data={}, source="TEST"))
        await bus.start()
        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 0

    def test_stats(self, bus):
        stats = bus.stats
        assert stats["events_processed"] == 0
        assert stats["errors"] == 0
