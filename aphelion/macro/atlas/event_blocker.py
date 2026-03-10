"""
ATLAS — Event Blocker
Phase 19 — Engineering Spec v3.0

Blocks trading around high-impact economic events.
Extends EconomicCalendar with automatic event ingestion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BlockWindow:
    """A window of time during which trading is blocked."""
    event_name: str
    start: datetime
    end: datetime
    impact: str = "HIGH"

    def is_active(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return self.start <= now <= self.end


class EventBlocker:
    """
    Prevents trading during high-impact economic events.

    Maintains a schedule of block windows and provides a simple
    is_blocked() check for the trading pipeline.
    """

    def __init__(
        self,
        pre_event_minutes: int = 30,
        post_event_minutes: int = 60,
    ):
        self._pre = timedelta(minutes=pre_event_minutes)
        self._post = timedelta(minutes=post_event_minutes)
        self._windows: List[BlockWindow] = []

    def add_event(self, name: str, event_time: datetime, impact: str = "HIGH") -> None:
        """Register an economic event to block around."""
        window = BlockWindow(
            event_name=name,
            start=event_time - self._pre,
            end=event_time + self._post,
            impact=impact,
        )
        self._windows.append(window)

    def is_blocked(self, now: Optional[datetime] = None) -> bool:
        """Check if trading is currently blocked."""
        now = now or datetime.now(timezone.utc)
        return any(w.is_active(now) for w in self._windows)

    def active_blocks(self, now: Optional[datetime] = None) -> List[BlockWindow]:
        """Return all currently active block windows."""
        now = now or datetime.now(timezone.utc)
        return [w for w in self._windows if w.is_active(now)]

    def next_block(self, now: Optional[datetime] = None) -> Optional[BlockWindow]:
        """Return the next upcoming block window (if any)."""
        now = now or datetime.now(timezone.utc)
        future = [w for w in self._windows if w.start > now]
        return min(future, key=lambda w: w.start) if future else None

    def cleanup_past(self, now: Optional[datetime] = None) -> int:
        """Remove expired block windows. Returns count removed."""
        now = now or datetime.now(timezone.utc)
        before = len(self._windows)
        self._windows = [w for w in self._windows if w.end > now]
        return before - len(self._windows)

    @property
    def window_count(self) -> int:
        return len(self._windows)
