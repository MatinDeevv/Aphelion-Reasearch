"""
NEXUS — Macro Signal Aggregator Core
Phase 19 — Engineering Spec v3.0

Aggregates signals from multiple macro sources into a unified score
for ARES consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MacroSignal:
    """A signal from a macro sub-module."""
    source: str
    direction: int      # -1, 0, 1
    confidence: float   # [0, 1]
    weight: float = 1.0


@dataclass
class NexusOutput:
    """Aggregated nexus output."""
    composite_score: float   # [-1, 1]
    signal_count: int
    agreement: float         # [0, 1] how much signals agree
    dominant_direction: int  # -1, 0, 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NexusCore:
    """
    Weighted aggregation of macro signals.
    Confidence-weighted voting across DXY, COT, sentiment, events.
    """

    def __init__(self):
        self._signals: List[MacroSignal] = []

    def add_signal(self, signal: MacroSignal) -> None:
        self._signals.append(signal)

    def clear(self) -> None:
        self._signals.clear()

    def aggregate(self) -> NexusOutput:
        """Produce a composite macro score."""
        if not self._signals:
            return NexusOutput(0.0, 0, 0.0, 0)

        weighted_sum = 0.0
        weight_total = 0.0
        for s in self._signals:
            weighted_sum += s.direction * s.confidence * s.weight
            weight_total += s.weight

        composite = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Agreement: proportion of signals pointing same direction as composite
        dominant = 1 if composite > 0.1 else (-1 if composite < -0.1 else 0)
        if dominant != 0:
            agreeing = sum(1 for s in self._signals if s.direction == dominant)
            agreement = agreeing / len(self._signals)
        else:
            agreement = 0.0

        return NexusOutput(
            composite_score=round(composite, 3),
            signal_count=len(self._signals),
            agreement=round(agreement, 3),
            dominant_direction=dominant,
        )
