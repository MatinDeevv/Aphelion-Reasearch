"""
SOLA — Veto Engine (Standalone)
Phase 21 — Engineering Spec v3.0

Standalone veto logic, separating veto decisions from the main SOLA class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VetoResult:
    """Result of a veto evaluation."""
    vetoed: bool
    reason: str = ""
    source: str = "SOLA"
    confidence_adjustment: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VetoEngine:
    """
    Encapsulates SOLA veto logic for reuse.

    Veto conditions:
    1. System in LOCKDOWN
    2. DEFENSIVE mode + low confidence
    3. Edge decay active
    4. Black swan detected
    5. Module health failure
    6. Event block active
    7. Regime awareness override
    """

    def __init__(self, defensive_min_confidence: float = 0.8):
        self._def_min_conf = defensive_min_confidence
        self._history: List[VetoResult] = []

    def evaluate(
        self,
        mode: str,
        edge_decay_active: bool,
        black_swan: bool,
        event_blocked: bool,
        confidence: float,
        module_healthy: bool = True,
        regime_tighten: bool = False,
    ) -> VetoResult:
        """Evaluate veto conditions."""
        # 1. Lockdown
        if mode == "LOCKDOWN":
            return self._veto("System LOCKDOWN")

        # 2. Black swan
        if black_swan:
            return self._veto("Black swan detected")

        # 3. Event block
        if event_blocked:
            return self._veto("Event block active")

        # 4. Edge decay
        if edge_decay_active:
            return self._veto("Edge decay active")

        # 5. Module health
        if not module_healthy:
            return self._veto("Source module unhealthy")

        # 6. Defensive mode filter
        min_conf = self._def_min_conf
        if regime_tighten:
            min_conf = min(0.95, min_conf + 0.1)

        if mode == "DEFENSIVE" and confidence < min_conf:
            return self._veto(
                f"DEFENSIVE: confidence {confidence:.2f} < {min_conf:.2f}",
                confidence_adjustment=confidence / min_conf,
            )

        return VetoResult(vetoed=False)

    def _veto(self, reason: str, confidence_adjustment: float = 0.0) -> VetoResult:
        result = VetoResult(vetoed=True, reason=reason, confidence_adjustment=confidence_adjustment)
        self._history.append(result)
        return result

    @property
    def veto_count(self) -> int:
        return len(self._history)

    @property
    def history(self) -> List[VetoResult]:
        return list(self._history)
