"""Tests for EventBus v2 improvements: history, latency, parallel dispatch."""

import asyncio
import pytest
from aphelion.core.event_bus import EventBus, Event, Priority
from aphelion.core.config import EventTopic


def _evt(topic=EventTopic.TICK, data=None, priority=Priority.NORMAL):
    """Helper to create events with required source field."""
    return Event(topic=topic, data=data or {}, source="TEST", priority=priority)


class TestEventBusHistory:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_history_empty_initially(self, bus):
        assert bus.get_history() == []

    @pytest.mark.asyncio
    async def test_history_captures_dispatched_events(self, bus):
        received = []

        async def cb(e):
            received.append(e)

        bus.subscribe(EventTopic.TICK, cb)
        event = _evt(EventTopic.TICK, {"price": 2850.0})
        await bus._dispatch(event)
        history = bus.get_history()
        assert len(history) >= 1
        assert history[-1].topic == EventTopic.TICK

    @pytest.mark.asyncio
    async def test_history_filter_by_topic(self, bus):
        async def noop(e):
            pass

        bus.subscribe(EventTopic.TICK, noop)
        bus.subscribe(EventTopic.SIGNAL, noop)
        await bus._dispatch(_evt(EventTopic.TICK))
        await bus._dispatch(_evt(EventTopic.SIGNAL))
        await bus._dispatch(_evt(EventTopic.TICK))

        tick_hist = bus.get_history(topic=EventTopic.TICK)
        assert all(e.topic == EventTopic.TICK for e in tick_hist)

    @pytest.mark.asyncio
    async def test_history_last_n(self, bus):
        async def noop(e):
            pass

        bus.subscribe(EventTopic.TICK, noop)
        for i in range(10):
            await bus._dispatch(_evt(EventTopic.TICK, {"i": i}))
        last3 = bus.get_history(last_n=3)
        assert len(last3) == 3


class TestEventBusLatency:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_avg_dispatch_ms_starts_zero(self, bus):
        assert bus.avg_dispatch_ms == 0.0

    @pytest.mark.asyncio
    async def test_avg_dispatch_ms_after_dispatch(self, bus):
        async def noop(e):
            pass

        bus.subscribe(EventTopic.TICK, noop)
        await bus._dispatch(_evt(EventTopic.TICK))
        assert bus.avg_dispatch_ms >= 0.0

    @pytest.mark.asyncio
    async def test_p99_dispatch_ms(self, bus):
        async def noop(e):
            pass

        bus.subscribe(EventTopic.TICK, noop)
        for _ in range(20):
            await bus._dispatch(_evt(EventTopic.TICK))
        assert bus.p99_dispatch_ms >= 0.0

    @pytest.mark.asyncio
    async def test_stats_includes_latency(self, bus):
        async def noop(e):
            pass

        bus.subscribe(EventTopic.TICK, noop)
        await bus._dispatch(_evt(EventTopic.TICK))
        stats = bus.stats
        assert "avg_dispatch_ms" in stats
        assert "p99_dispatch_ms" in stats
        assert "history_size" in stats


class TestEventBusParallelDispatch:
    @pytest.mark.asyncio
    async def test_normal_priority_callbacks_run(self):
        bus = EventBus()
        results = []

        async def cb1(e):
            results.append("cb1")

        async def cb2(e):
            results.append("cb2")

        bus.subscribe(EventTopic.TICK, cb1)
        bus.subscribe(EventTopic.TICK, cb2)
        # NORMAL priority -> parallel dispatch via asyncio.gather
        await bus._dispatch(_evt(EventTopic.TICK, priority=Priority.NORMAL))
        assert "cb1" in results
        assert "cb2" in results

    @pytest.mark.asyncio
    async def test_critical_callbacks_run_sequentially(self):
        bus = EventBus()
        order = []

        async def first(e):
            order.append(1)

        async def second(e):
            order.append(2)

        bus.subscribe(EventTopic.SIGNAL, first)
        bus.subscribe(EventTopic.SIGNAL, second)
        # CRITICAL priority -> sequential dispatch
        await bus._dispatch(_evt(EventTopic.SIGNAL, priority=Priority.CRITICAL))
        assert order == [1, 2]
