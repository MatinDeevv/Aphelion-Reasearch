"""
APHELION Component Registry
Tracks all module health, status, and resource allocation.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from aphelion.core.config import ComponentStatus, ModuleInfo, MODULES, Tier


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


class Registry:
    """Central registry for all APHELION components."""

    def __init__(self):
        self._components: dict[str, ComponentState] = {}
        self._start_time = time.time()

    def register(self, name: str) -> None:
        if name not in MODULES:
            raise ValueError(f"Unknown module: {name}")
        self._components[name] = ComponentState(
            info=MODULES[name],
            status=ComponentStatus.INITIALIZING,
        )

    def set_status(self, name: str, status: ComponentStatus) -> None:
        self._ensure_registered(name)
        self._components[name].status = status

    def heartbeat(self, name: str) -> None:
        self._ensure_registered(name)
        self._components[name].last_heartbeat = time.time()

    def report_error(self, name: str, error: str) -> None:
        self._ensure_registered(name)
        comp = self._components[name]
        comp.error_count += 1
        comp.last_error = error
        if comp.error_count >= 10:
            comp.status = ComponentStatus.ERROR

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

    def system_health(self) -> dict:
        active = self.get_active_components()
        if not active:
            return {"overall": 0.0, "active_count": 0, "total_count": len(self._components)}

        avg_health = sum(s.health_score for s in active.values()) / len(active)
        return {
            "overall": avg_health,
            "active_count": len(active),
            "total_count": len(self._components),
            "uptime_seconds": time.time() - self._start_time,
        }

    def _ensure_registered(self, name: str) -> None:
        if name not in self._components:
            raise KeyError(f"Component '{name}' not registered. Call register() first.")
