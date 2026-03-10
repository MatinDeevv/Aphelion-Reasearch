"""
SOLA — Improvement Loop
Phase 21 — Engineering Spec v3.0

Continuous self-improvement: evaluates module performance,
identifies weaknesses, and triggers optimization/retraining.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ImprovementAction:
    """A recommended improvement action."""
    target_module: str
    action: str          # "RETRAIN", "REOPTIMIZE", "RESTART", "REVIEW_PARAMS"
    reason: str
    priority: int = 1    # 1 = highest
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ImprovementLoop:
    """
    SOLA self-improvement: runs periodic cycles that:
    1. Collect module performance metrics
    2. Rank modules by contribution
    3. Identify under-performers
    4. Generate improvement actions
    5. Track action history
    """

    def __init__(self):
        self._cycle_count: int = 0
        self._module_scores: Dict[str, List[float]] = {}
        self._action_history: List[ImprovementAction] = []

    def record_contribution(self, module: str, score: float) -> None:
        """Record a module's contribution to a trade outcome."""
        self._module_scores.setdefault(module, []).append(score)

    def run_cycle(self) -> List[ImprovementAction]:
        """Run one improvement cycle. Returns recommended actions."""
        self._cycle_count += 1
        actions = []

        for module, scores in self._module_scores.items():
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            recent = scores[-20:] if len(scores) >= 20 else scores
            recent_avg = sum(recent) / len(recent)

            # Degrading performance
            if len(scores) >= 50 and recent_avg < avg * 0.8:
                actions.append(ImprovementAction(
                    target_module=module,
                    action="RETRAIN",
                    reason=f"Recent avg {recent_avg:.3f} < 80% of overall {avg:.3f}",
                    priority=1,
                ))
            # Negative contributor
            elif avg < 0:
                actions.append(ImprovementAction(
                    target_module=module,
                    action="REVIEW_PARAMS",
                    reason=f"Negative avg contribution {avg:.3f}",
                    priority=2,
                ))

        self._action_history.extend(actions)
        return actions

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def action_history(self) -> List[ImprovementAction]:
        return list(self._action_history)

    def get_module_avg(self, module: str) -> float:
        scores = self._module_scores.get(module, [])
        return sum(scores) / len(scores) if scores else 0.0
