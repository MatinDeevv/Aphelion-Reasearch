"""
APHELION SENTINEL Monitor
Async background task that polls open positions every 100ms.
Publishes CRITICAL risk events when stop-loss levels are breached.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aphelion.core.config import EventTopic
from aphelion.core.event_bus import EventBus, Event, Priority

if TYPE_CHECKING:
    from aphelion.risk.sentinel.core import SentinelCore  # noqa: F401


class SentinelMonitor:
    """Monitors open positions for stop-loss breaches in real time."""

    def __init__(self, event_bus: EventBus, sentinel_core: "SentinelCore"):
        self._event_bus = event_bus
        self._sentinel = sentinel_core
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._last_price: float = 0.0
        self._sl_breach_count: int = 0
        self._check_interval_ms: int = 100  # 100ms polling cadence

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Price feed ────────────────────────────────────────────────────────────

    def update_price(self, price: float) -> None:
        self._last_price = price

    # ── Internal loop ─────────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await self._check_positions()
                await self._check_friday_close()
                await asyncio.sleep(self._check_interval_ms / 1000.0)
            except asyncio.CancelledError:
                break

    async def _check_positions(self) -> None:
        """Check every open position for stop-loss breach at current price."""
        positions = self._sentinel.get_open_positions()
        for position in positions:
            breached = False
            if position.direction == "LONG" and self._last_price <= position.stop_loss:
                breached = True
            elif position.direction == "SHORT" and self._last_price >= position.stop_loss:
                breached = True

            if breached:
                self._sl_breach_count += 1
                self._event_bus.publish_nowait(Event(
                    topic=EventTopic.RISK,
                    data={
                        "action": "SL_BREACH",
                        "position_id": position.position_id,
                        "entry": position.entry_price,
                        "stop_loss": position.stop_loss,
                        "current_price": self._last_price,
                        "direction": position.direction,
                    },
                    source="SENTINEL",
                    priority=Priority.CRITICAL,
                ))

    async def _check_friday_close(self) -> None:
        """Force-close all positions 30 min before Friday 21:00 UTC close (Section 6.4)."""
        if not hasattr(self._sentinel, '_clock'):
            return
        clock = self._sentinel._clock
        if clock is None:
            return
        if not clock.is_friday_lockout():
            return
        positions = self._sentinel.get_open_positions()
        for position in positions:
            self._event_bus.publish_nowait(Event(
                topic=EventTopic.RISK,
                data={
                    "action": "FRIDAY_CLOSE",
                    "position_id": position.position_id,
                    "entry": position.entry_price,
                    "current_price": self._last_price,
                    "direction": position.direction,
                    "reason": "Friday close lockout — 30 min before market close",
                },
                source="SENTINEL",
                priority=Priority.CRITICAL,
            ))

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "last_price": self._last_price,
            "sl_breach_count": self._sl_breach_count,
            "check_interval_ms": self._check_interval_ms,
        }
