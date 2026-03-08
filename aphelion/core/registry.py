"""
APHELION Component Registry
Tracks all module health, status, and resource allocation.
Supports heartbeat timeout detection, deregistration, and min-health system scoring.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from aphelion.core.config import ComponentStatus, ModuleInfo, MODULES, Tier

logger = logging.getLogger(__name__)

HEARTBEAT_TIMEOUT_SECONDS = 30.0  # mark stale after 30s without heartbeat
MAX_ERRORS_BEFORE_FAULT = 10


@dataclass
class ComponentState:
    info: ModuleInfo
    status: ComponentStatus = ComponentStatus.DISABLED
    health_score: float = 100.0
    cpu_cores: list[int] = field(default_factory=list)
    gpu_vram_gb: float = 0.0
    last_heartbeat: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    registered_at: float = 0.0


class Registry:
    """Central registry for all APHELION components.

    Improvements over Phase 1:
    - Heartbeat timeout detection (stale components auto-flagged)
    - Deregister support
    - Min-health-based system scoring (min of active, not just average)
    - Stale component listing
    """

    def __init__(self, heartbeat_timeout: float = HEARTBEAT_TIMEOUT_SECONDS):
        self._components: dict[str, ComponentState] = {}
        self._start_time = time.time()
        self._heartbeat_timeout = heartbeat_timeout

    def register(self, name: str) -> None:
        if name not in MODULES:
            raise ValueError(f"Unknown module: {name}")
        self._components[name] = ComponentState(
            info=MODULES[name],
            status=ComponentStatus.INITIALIZING,
            registered_at=time.time(),
        )

    def deregister(self, name: str) -> None:
        """Remove a component from the registry."""
        self._ensure_registered(name)
        del self._components[name]
        logger.info("Deregistered component: %s", name)

    def set_status(self, name: str, status: ComponentStatus) -> None:
        self._ensure_registered(name)
        old = self._components[name].status
        self._components[name].status = status
        if old != status:
            logger.debug("Component %s: %s -> %s", name, old.name, status.name)

    def heartbeat(self, name: str) -> None:
        self._ensure_registered(name)
        self._components[name].last_heartbeat = time.time()

    def report_error(self, name: str, error: str) -> None:
        self._ensure_registered(name)
        comp = self._components[name]
        comp.error_count += 1
        comp.last_error = error
        if comp.error_count >= MAX_ERRORS_BEFORE_FAULT:
            comp.status = ComponentStatus.ERROR
            logger.warning("Component %s exceeded error threshold (%d errors)", name, comp.error_count)

    def set_health(self, name: str, score: float) -> None:
        self._ensure_registered(name)
        self._components[name].health_score = max(0.0, min(100.0, score))

    def allocate_cpu(self, name: str, cores: list[int]) -> None:
        self._ensure_registered(name)
        self._components[name].cpu_cores = cores

    def allocate_gpu(self, name: str, vram_gb: float) -> None:
        self._ensure_registered(name)
        self._components[name].gpu_vram_gb = vram_gb

    def get_status(self, name: str) -> ComponentState:
        self._ensure_registered(name)
        return self._components[name]

    def get_active_components(self) -> dict[str, ComponentState]:
        return {
            name: state for name, state in self._components.items()
            if state.status == ComponentStatus.ACTIVE
        }

    def get_components_by_tier(self, tier: Tier) -> dict[str, ComponentState]:
        return {
            name: state for name, state in self._components.items()
            if state.info.tier == tier
        }

    def get_stale_components(self) -> dict[str, ComponentState]:
        """Return components whose heartbeat has timed out."""
        now = time.time()
        stale = {}
        for name, state in self._components.items():
            if state.status == ComponentStatus.ACTIVE and state.last_heartbeat > 0:
                if (now - state.last_heartbeat) > self._heartbeat_timeout:
                    stale[name] = state
        return stale

    def check_heartbeats(self) -> list[str]:
        """Check for stale heartbeats and mark those components. Returns list of stale names."""
        stale = self.get_stale_components()
        stale_names = []
        for name in stale:
            self._components[name].status = ComponentStatus.ERROR
            logger.warning("Component %s heartbeat timed out (>%.0fs)", name, self._heartbeat_timeout)
            stale_names.append(name)
        return stale_names

    def system_health(self) -> dict:
        active = self.get_active_components()
        if not active:
            return {"overall": 0.0, "min_health": 0.0, "active_count": 0, "total_count": len(self._components)}

        health_scores = [s.health_score for s in active.values()]
        avg_health = sum(health_scores) / len(health_scores)
        min_health = min(health_scores)
        # Weighted: 70% average + 30% min (weakest link awareness)
        overall = 0.7 * avg_health + 0.3 * min_health
        return {
            "overall": overall,
            "avg_health": avg_health,
            "min_health": min_health,
            "active_count": len(active),
            "total_count": len(self._components),
            "stale_count": len(self.get_stale_components()),
            "uptime_seconds": time.time() - self._start_time,
        }

    def _ensure_registered(self, name: str) -> None:
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered. Call register() first.")
