"""
MERIDIAN — Cross-Module State Synchronization
Phase 16 — Engineering Spec v3.0

Ensures all modules share a consistent view of system state.
On crash, MERIDIAN can restore system to last known good state within 5 seconds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import json
import os
import copy


@dataclass
class StateSnapshot:
    """Point-in-time system state snapshot."""
    snapshot_id: str
    timestamp: datetime
    label: str
    state: Dict[str, Any]
    modules_active: List[str] = field(default_factory=list)


class StateBus:
    """Central state publisher/subscriber for cross-module communication."""

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._snapshots: List[StateSnapshot] = []
        self._snapshot_counter = 0

    def publish(self, key: str, value: Any) -> None:
        """Publish a state update. Notifies all subscribers of this key."""
        self._state[key] = value
        for callback in self._subscribers.get(key, []):
            try:
                callback(key, value)
            except Exception:
                pass  # Don't let subscriber errors break the bus

    def subscribe(self, key: str, callback: Callable) -> None:
        """Subscribe to state changes for a key."""
        self._subscribers.setdefault(key, []).append(callback)

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._state)

    def create_snapshot(self, label: str = "") -> StateSnapshot:
        """Create a point-in-time snapshot of all state."""
        self._snapshot_counter += 1
        snapshot = StateSnapshot(
            snapshot_id=f"SNAP_{self._snapshot_counter:06d}",
            timestamp=datetime.now(timezone.utc),
            label=label,
            state=copy.deepcopy(self._state),
            modules_active=list(self._subscribers.keys()),
        )
        self._snapshots.append(snapshot)
        # Keep only last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]
        return snapshot

    def restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore system state from a snapshot."""
        self._state = copy.deepcopy(snapshot.state)
        # Notify all subscribers of restored state
        for key, value in self._state.items():
            for callback in self._subscribers.get(key, []):
                try:
                    callback(key, value)
                except Exception:
                    pass

    def get_latest_snapshot(self) -> Optional[StateSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def clear(self) -> None:
        self._state.clear()


class StateRecovery:
    """Recovery manager for crash scenarios."""

    def __init__(self, state_bus: StateBus, recovery_dir: str = "data/state"):
        self._bus = state_bus
        self._dir = recovery_dir

    def persist_state(self) -> str:
        """Persist current state to disk for crash recovery."""
        os.makedirs(self._dir, exist_ok=True)
        snapshot = self._bus.create_snapshot(label="persist")
        path = os.path.join(self._dir, f"{snapshot.snapshot_id}.json")

        serializable_state = {}
        for key, value in snapshot.state.items():
            try:
                json.dumps(value)
                serializable_state[key] = value
            except (TypeError, ValueError):
                serializable_state[key] = str(value)

        with open(path, 'w') as f:
            json.dump({
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "label": snapshot.label,
                "state": serializable_state,
            }, f, indent=2)
        return path

    def recover_latest(self) -> bool:
        """Recover from the latest persisted state."""
        if not os.path.exists(self._dir):
            return False

        files = sorted(
            [f for f in os.listdir(self._dir) if f.endswith('.json')],
            reverse=True,
        )
        if not files:
            return False

        path = os.path.join(self._dir, files[0])
        with open(path) as f:
            data = json.load(f)

        for key, value in data.get("state", {}).items():
            self._bus.publish(key, value)

        return True
