"""
APHELION Event Bus
Async pub/sub message passing for inter-module communication.
Dedicated core C03. Priority handling for SENTINEL risk events.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine

from aphelion.core.config import EventTopic


class Priority(IntEnum):
    CRITICAL = 0    # SENTINEL kill switches
    HIGH = 1        # Risk events, signals
    NORMAL = 2      # Standard messages
    LOW = 3         # Health checks, logs


@dataclass
class Event:
    topic: EventTopic
    data: Any
    source: str
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: "Event") -> bool:
        return (self.priority, self.timestamp) < (other.priority, other.timestamp)


Callback = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async pub/sub event bus with priority queues."""

    def __init__(self, max_queue_size: int = 10_000):
        self._subscribers: dict[EventTopic, list[Callback]] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._running = False
        self._task: asyncio.Task | None = None
        self._event_count = 0
        self._error_count = 0

    def subscribe(self, topic: EventTopic, callback: Callback) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: EventTopic, callback: Callback) -> None:
        if topic in self._subscribers:
            self._subscribers[topic] = [
                cb for cb in self._subscribers[topic] if cb is not callback
            ]

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop lowest priority events when full — SENTINEL events never dropped
            if event.priority == Priority.CRITICAL:
                # Force it in — critical events must never be lost
                self._queue._queue.append(event)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
                self._event_count += 1
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _dispatch(self, event: Event) -> None:
        callbacks = self._subscribers.get(event.topic, [])
        for callback in callbacks:
            try:
                await callback(event)
            except Exception:
                self._error_count += 1

    @property
    def stats(self) -> dict:
        return {
            "events_processed": self._event_count,
            "errors": self._error_count,
            "queue_size": self._queue.qsize(),
            "subscribers": {
                topic.value: len(cbs) for topic, cbs in self._subscribers.items()
            },
        }
