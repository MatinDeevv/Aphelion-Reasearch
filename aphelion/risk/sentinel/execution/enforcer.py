"""
APHELION SENTINEL Execution Enforcer
Last line of defense before any order reaches the broker.
Applies circuit-breaker size multiplier then re-validates every proposal.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from aphelion.risk.sentinel.validator import TradeValidator, TradeProposal  # noqa: F401


class ExecutionEnforcer:
    """
    Wraps every outbound order and checks it one final time.
    If anything fails the order is rejected and logged.
    """

    def __init__(self, validator: "TradeValidator", circuit_breaker: CircuitBreaker):
        self._validator = validator
        self._cb = circuit_breaker
        self._approved_count: int = 0
        self._rejected_count: int = 0
        self._rejection_log: deque = deque(maxlen=1000)

    # ── Order gate ────────────────────────────────────────────────────────────

    def approve_order(self, proposal: "TradeProposal") -> tuple[bool, str, float]:
        """
        Apply CB multiplier, re-validate, and return (approved, reason, final_size_pct).
        """
        # Step 1: Apply circuit-breaker size multiplier
        adjusted_size = self._cb.apply_multiplier(proposal.size_pct)
        adjusted_proposal = proposal.with_size(adjusted_size)

        # Step 2: Run full validator
        result = self._validator.validate(adjusted_proposal)

        # Step 3: Return outcome
        if result.approved:
            self._approved_count += 1
            return (True, "APPROVED", result.adjusted_size_pct)
        else:
            self._rejected_count += 1
            reason = " | ".join(result.rejections)
            self._rejection_log.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "proposed_by": proposal.proposed_by,
                "reason": reason,
            })
            return (False, reason, 0.0)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_rejection_summary(self) -> dict:
        total = self._approved_count + self._rejected_count
        rejection_rate = self._rejected_count / total if total > 0 else 0.0
        return {
            "approved_count": self._approved_count,
            "rejected_count": self._rejected_count,
            "rejection_rate": rejection_rate,
            "recent_rejections": list(self._rejection_log)[-10:],
        }

    @property
    def stats(self) -> dict:
        return self.get_rejection_summary()
