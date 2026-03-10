"""
NEMESIS VERDICT — Final Trade Verdict Engine
Phase 14 — Provides final go/no-go on every trade based on NEMESIS analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    """NEMESIS final verdict on a proposed trade."""
    approved: bool
    confidence_adjustment: float   # Multiplier on confidence [0, 1.2]
    reason: str
    stress_level: str             # "LOW", "MODERATE", "HIGH", "CRITICAL"
    warnings: List[str]


class VerdictCore:
    """
    Synthesizes all NEMESIS sub-module signals into a final verdict.
    Can veto, adjust confidence, or flag warnings on any proposed trade.
    """

    def __init__(
        self,
        stress_veto_threshold: float = 0.8,
        stress_reduce_threshold: float = 0.5,
    ):
        self._stress_veto = stress_veto_threshold
        self._stress_reduce = stress_reduce_threshold

    def evaluate(
        self,
        stress_score: float,
        contrarian_active: bool = False,
        temporal_anomaly: bool = False,
        tail_event: bool = False,
        overfitting_detected: bool = False,
    ) -> Verdict:
        """Evaluate all NEMESIS signals and produce a verdict."""
        warnings: List[str] = []
        confidence_adj = 1.0

        # Determine stress level
        if stress_score > self._stress_veto:
            stress_level = "CRITICAL"
        elif stress_score > self._stress_reduce:
            stress_level = "HIGH"
        elif stress_score > 0.3:
            stress_level = "MODERATE"
        else:
            stress_level = "LOW"

        # Veto conditions
        if stress_score > self._stress_veto:
            return Verdict(
                approved=False,
                confidence_adjustment=0.0,
                reason="STRESS_CRITICAL",
                stress_level=stress_level,
                warnings=["System stress critical — all trades vetoed"],
            )

        if tail_event:
            return Verdict(
                approved=False,
                confidence_adjustment=0.0,
                reason="TAIL_EVENT_ACTIVE",
                stress_level=stress_level,
                warnings=["Tail risk event detected — trading halted"],
            )

        # Adjustments
        if stress_score > self._stress_reduce:
            confidence_adj *= 0.5
            warnings.append(f"High stress ({stress_score:.0%}) — confidence halved")

        if contrarian_active:
            confidence_adj *= 0.7
            warnings.append("NEMESIS contrarian active — reduced confidence")

        if temporal_anomaly:
            confidence_adj *= 0.8
            warnings.append("Temporal anomaly detected")

        if overfitting_detected:
            confidence_adj *= 0.6
            warnings.append("Overfitting detected — reduced confidence")

        return Verdict(
            approved=True,
            confidence_adjustment=confidence_adj,
            reason="APPROVED" if not warnings else "APPROVED_WITH_WARNINGS",
            stress_level=stress_level,
            warnings=warnings,
        )
