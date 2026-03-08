"""
APHELION SENTINEL Circuit Breaker
Real-time drawdown monitoring with 3 escalating alert levels.
L1 (3%) → reduce size 50%
L2 (6%) → halt new trades (close-only mode)
L3 (10%) → full halt, close all positions

Improvements:
- Uses SENTINEL config thresholds instead of hardcoded values
- Cooldown period after L2 recovery (prevents rapid re-entry)
- L2 recovery path with hysteresis
- Configurable thresholds via constructor
"""

import time
from datetime import datetime, timezone

from aphelion.core.config import SENTINEL, EventTopic
from aphelion.core.event_bus import EventBus, Event, Priority


class CircuitBreaker:
    """Monitors account equity drawdown and enforces progressive risk responses."""

    def __init__(
        self,
        event_bus: EventBus,
        l1_threshold: float | None = None,
        l2_threshold: float | None = None,
        l3_threshold: float | None = None,
        cooldown_seconds: float = 300.0,  # 5 min cooldown after recovery
    ):
        self._event_bus = event_bus
        self._state: str = "NORMAL"
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._triggers: list[dict] = []
        self._size_multiplier: float = 1.0
        self._cooldown_seconds = cooldown_seconds
        self._last_recovery_time: float = 0.0  # timestamp of last recovery from L1/L2

        # Use SENTINEL config values as defaults
        self.L1_THRESHOLD = l1_threshold if l1_threshold is not None else SENTINEL.daily_equity_drawdown_l1
        self.L2_THRESHOLD = l2_threshold if l2_threshold is not None else SENTINEL.daily_equity_drawdown_l2
        self.L3_THRESHOLD = l3_threshold if l3_threshold is not None else SENTINEL.daily_equity_drawdown_l3

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, equity: float) -> str:
        """Update equity, evaluate drawdown level, return current state."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        dd = self.current_drawdown

        if dd >= self.L3_THRESHOLD and self._state != "L3":
            self.trigger_l3(dd)
        elif dd >= self.L2_THRESHOLD and self._state not in ("L2", "L3"):
            self.trigger_l2(dd)
        elif dd >= self.L1_THRESHOLD and self._state == "NORMAL":
            self.trigger_l1(dd)
        elif dd < self.L1_THRESHOLD and self._state == "L1":
            self.reset()
        elif dd < self.L2_THRESHOLD and self._state == "L2":
            # L2 recovery with hysteresis: recover to L1 first, not NORMAL
            self._recover_from_l2(dd)

        return self._state

    def trigger_l1(self, drawdown: float) -> None:
        self._state = "L1"
        self._size_multiplier = 0.50
        self._triggers.append({
            "level": "L1",
            "drawdown": drawdown,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        self._event_bus.publish_nowait(Event(
            topic=EventTopic.RISK,
            data={"level": "L1", "drawdown": drawdown, "action": "REDUCE_SIZE_50PCT"},
            source="SENTINEL",
            priority=Priority.HIGH,
        ))

    def trigger_l2(self, drawdown: float) -> None:
        self._state = "L2"
        self._size_multiplier = 0.0  # No new trades
        self._triggers.append({
            "level": "L2",
            "drawdown": drawdown,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        self._event_bus.publish_nowait(Event(
            topic=EventTopic.RISK,
            data={"level": "L2", "drawdown": drawdown, "action": "HALT_NO_NEW_TRADES"},
            source="SENTINEL",
            priority=Priority.HIGH,
        ))

    def trigger_l3(self, drawdown: float) -> None:
        self._state = "L3"
        self._size_multiplier = 0.0
        self._triggers.append({
            "level": "L3",
            "drawdown": drawdown,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        self._event_bus.publish_nowait(Event(
            topic=EventTopic.RISK,
            data={"level": "L3", "drawdown": drawdown, "action": "FULL_HALT_CLOSE_ALL"},
            source="SENTINEL",
            priority=Priority.CRITICAL,
        ))

    def _recover_from_l2(self, drawdown: float) -> None:
        """Recover from L2 to L1 (with cooldown check)."""
        now = time.time()
        if self._last_recovery_time > 0:
            elapsed = now - self._last_recovery_time
            if elapsed < self._cooldown_seconds:
                return  # Still in cooldown, stay at L2
        self._state = "L1"
        self._size_multiplier = 0.50
        self._last_recovery_time = now
        self._triggers.append({
            "level": "L2_RECOVERY",
            "drawdown": drawdown,
            "time": datetime.now(timezone.utc).isoformat(),
        })

    def reset(self) -> None:
        """Reset from L1 back to NORMAL. Only valid when state is L1."""
        if self._state != "L1":
            return
        self._state = "NORMAL"
        self._size_multiplier = 1.0
        self._last_recovery_time = time.time()

    def apply_multiplier(self, proposed_size_pct: float) -> float:
        """Apply circuit breaker multiplier. Clamps result to [0.0, max_position_pct]."""
        result = proposed_size_pct * self._size_multiplier
        return max(0.0, min(result, SENTINEL.max_position_pct))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def size_multiplier(self) -> float:
        return self._size_multiplier

    @property
    def current_drawdown(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def get_summary(self) -> dict:
        return {
            "state": self._state,
            "size_multiplier": self._size_multiplier,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "trigger_count": len(self._triggers),
            "trigger_history": self._triggers[-10:],
        }
