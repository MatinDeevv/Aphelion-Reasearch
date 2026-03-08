"""
APHELION Event Bus
Async pub/sub message passing for inter-module communication.
Dedicated core C03. Priority handling for SENTINEL risk events.
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine

from aphelion.core.config import EventTopic

logger = logging.getLogger(__name__)

# Sentinel topic for wildcard subscribers that receive ALL events
_WILDCARD = object()


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
        self._dropped_count = 0

    def subscribe(self, topic: EventTopic, callback: Callback) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def subscribe_all(self, callback: Callback) -> None:
        """Subscribe to ALL event topics (wildcard). Useful for audit logging."""
        if _WILDCARD not in self._subscribers:
            self._subscribers[_WILDCARD] = []
        self._subscribers[_WILDCARD].append(callback)

    def unsubscribe(self, topic: EventTopic, callback: Callback) -> None:
        if topic in self._subscribers:
            self._subscribers[topic] = [
                cb for cb in self._subscribers[topic] if cb is not callback
            ]

    def unsubscribe_all(self, topic: EventTopic) -> None:
        """Remove all subscribers for a topic."""
        self._subscribers.pop(topic, None)

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # CRITICAL events must never be lost — force into the heap safely
            if event.priority == Priority.CRITICAL:
                heapq.heappush(self._queue._queue, event)
                logger.warning("Queue full — forced CRITICAL event: %s", event.topic)
            else:
                self._dropped_count += 1
                logger.warning("Queue full — dropped %s event from %s", event.topic, event.source)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())

    async def stop(self, drain: bool = True) -> None:
        """Stop the event bus. If drain=True, process remaining events before stopping."""
        self._running = False
        if drain:
            while not self._queue.empty():
                try:
                    event = self._queue.get_nowait()
                    await self._dispatch(event)
                    self._event_count += 1
                except asyncio.QueueEmpty:
                    break
                except Exception:
                    self._error_count += 1
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
        # Also dispatch to wildcard subscribers
        wildcard_cbs = self._subscribers.get(_WILDCARD, [])
        for callback in (*callbacks, *wildcard_cbs):
            try:
                await callback(event)
            except Exception as exc:
                self._error_count += 1
                logger.error(
                    "EventBus dispatch error on %s from %s: %s",
                    event.topic, event.source, exc,
                )

    @property
    def stats(self) -> dict:
        return {
            "events_processed": self._event_count,
            "errors": self._error_count,
            "dropped": self._dropped_count,
            "queue_size": self._queue.qsize(),
            "subscribers": {
                (topic.value if hasattr(topic, 'value') else '__wildcard__'): len(cbs)
                for topic, cbs in self._subscribers.items()
            },
        }
