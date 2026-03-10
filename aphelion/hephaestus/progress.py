"""
HEPHAESTUS — Forge Progress Streaming (Phase 23)

Streams forge progress messages to the TUI in real-time.
The HEPHAESTUS agent calls ``emit()`` at each pipeline stage.
The TUI polls ``get_latest()`` to update the progress bar.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ForgeUpdate:
    """A single progress update from the forge pipeline."""
    stage: str
    message: str
    percent: float
    timestamp: float = field(default_factory=time.monotonic)
    details: dict = field(default_factory=dict)


class ForgeProgressStream:
    """Thread-safe progress stream for HEPHAESTUS → TUI communication.

    The forge agent calls ``emit()`` from a background thread.
    The TUI polls ``get_latest()`` from the main thread.
    """

    def __init__(self) -> None:
        self._updates: list[ForgeUpdate] = []
        self._lock = threading.Lock()
        self._complete = False
        self._success = False

    def emit(self, stage: str, message: str, pct: float, **details: object) -> None:
        """Emit a progress update (called from forge thread)."""
        update = ForgeUpdate(
            stage=stage,
            message=message,
            percent=max(0.0, min(1.0, pct)),
            details=dict(details),
        )
        with self._lock:
            self._updates.append(update)

    def mark_complete(self, success: bool, message: str = "") -> None:
        """Mark the forge as complete."""
        with self._lock:
            self._complete = True
            self._success = success
            if message:
                self._updates.append(ForgeUpdate(
                    stage="complete",
                    message=message,
                    percent=1.0,
                ))

    def get_latest(self) -> list[ForgeUpdate]:
        """Return all updates since last call (called from TUI thread)."""
        with self._lock:
            updates = list(self._updates)
            self._updates.clear()
            return updates

    @property
    def is_complete(self) -> bool:
        with self._lock:
            return self._complete

    @property
    def was_successful(self) -> bool:
        with self._lock:
            return self._success

    def reset(self) -> None:
        """Reset for a new forge operation."""
        with self._lock:
            self._updates.clear()
            self._complete = False
            self._success = False
