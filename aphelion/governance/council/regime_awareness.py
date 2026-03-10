"""
SOLA — Regime Awareness
Phase 21 — Engineering Spec v3.0

Makes SOLA aware of the current market regime so it can
adjust its mode transitions and veto thresholds accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RegimeContext:
    """Current regime context for SOLA decision-making."""
    regime: str = "UNKNOWN"         # TRENDING, RANGING, VOLATILE, CRISIS
    volatility_percentile: float = 0.5
    session: str = "UNKNOWN"        # LONDON, NEWYORK, ASIA, OVERLAP
    dxy_bias: int = 0               # -1, 0, 1
    event_blocked: bool = False


class RegimeAwareness:
    """
    Provides regime-aware adjustments to SOLA parameters.

    In CRISIS regime: lower edge confidence threshold, tighter veto
    In TRENDING: standard thresholds
    In RANGING: slightly tighter (mean-reversion traps)
    """

    REGIME_MULTIPLIERS: Dict[str, float] = {
        "TRENDING": 1.0,
        "RANGING": 0.9,
        "VOLATILE": 0.75,
        "CRISIS": 0.5,
        "UNKNOWN": 0.85,
    }

    def __init__(self):
        self._context = RegimeContext()

    def update(
        self,
        regime: str = "UNKNOWN",
        volatility_percentile: float = 0.5,
        session: str = "UNKNOWN",
        dxy_bias: int = 0,
        event_blocked: bool = False,
    ) -> RegimeContext:
        self._context = RegimeContext(
            regime=regime,
            volatility_percentile=volatility_percentile,
            session=session,
            dxy_bias=dxy_bias,
            event_blocked=event_blocked,
        )
        return self._context

    def confidence_multiplier(self) -> float:
        """Regime-based confidence multiplier for edge thresholds."""
        return self.REGIME_MULTIPLIERS.get(self._context.regime, 0.85)

    def should_tighten_veto(self) -> bool:
        """True if regime warrants tighter veto thresholds."""
        return self._context.regime in ("VOLATILE", "CRISIS") or self._context.event_blocked

    @property
    def context(self) -> RegimeContext:
        return self._context
