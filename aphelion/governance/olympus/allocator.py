"""
OLYMPUS — Capital Allocator
Phase 20 — Engineering Spec v3.0

Dynamic capital allocation between ALPHA and OMEGA strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Allocation:
    """Capital allocation state."""
    alpha_pct: float = 0.70
    omega_pct: float = 0.30
    alpha_max_risk_pct: float = 1.0
    omega_max_risk_pct: float = 2.0


class CapitalAllocator:
    """
    Allocates capital between ALPHA (M1 scalp) and OMEGA (H1/H4 swing)
    based on relative performance and regime context.
    """

    MIN_PCT = 0.10
    MAX_PCT = 0.90

    def __init__(self, initial_alpha: float = 0.70):
        self._allocation = Allocation(
            alpha_pct=initial_alpha,
            omega_pct=1.0 - initial_alpha,
        )
        self._alpha_sharpe: float = 0.0
        self._omega_sharpe: float = 0.0

    def update_sharpes(self, alpha_sharpe: float, omega_sharpe: float) -> None:
        self._alpha_sharpe = alpha_sharpe
        self._omega_sharpe = omega_sharpe

    def rebalance(self) -> Allocation:
        """Sharpe-weighted rebalance."""
        a = max(0.1, self._alpha_sharpe)
        o = max(0.1, self._omega_sharpe)
        total = a + o

        alpha_pct = max(self.MIN_PCT, min(self.MAX_PCT, a / total))
        omega_pct = 1.0 - alpha_pct

        self._allocation = Allocation(
            alpha_pct=round(alpha_pct, 3),
            omega_pct=round(omega_pct, 3),
        )
        return self._allocation

    @property
    def allocation(self) -> Allocation:
        return self._allocation
