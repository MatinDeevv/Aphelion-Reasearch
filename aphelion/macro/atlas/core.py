"""
ATLAS — Macro Intelligence Core
Phase 19 — Engineering Spec v3.0

Aggregates all macro feeds (DXY, COT, events, sentiment) into a
single MacroContext for ARES consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MacroContext:
    """Aggregated macro context for ARES."""
    dxy_bias: int = 0            # -1 bearish gold, 0 neutral, 1 bullish gold
    cot_bias: str = "NEUTRAL"    # EXTREME_LONG, BULLISH, NEUTRAL, BEARISH, EXTREME_SHORT
    event_blocked: bool = False
    sentiment: float = 0.0       # [-1, 1]
    macro_score: float = 0.0     # [-1, 1] composite
    freshness: str = "EXPIRED"   # FRESH, STALE, EXPIRED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AtlasCore:
    """
    Central aggregator for all macro feeds.
    Produces a single MacroContext with a composite score.
    """

    def __init__(self):
        self._dxy_bias: int = 0
        self._cot_bias: str = "NEUTRAL"
        self._event_blocked: bool = False
        self._sentiment: float = 0.0
        self._freshness: str = "EXPIRED"

    def update_dxy(self, bias: int) -> None:
        self._dxy_bias = bias

    def update_cot(self, bias: str) -> None:
        self._cot_bias = bias

    def update_event_block(self, blocked: bool) -> None:
        self._event_blocked = blocked

    def update_sentiment(self, score: float) -> None:
        self._sentiment = max(-1.0, min(1.0, score))

    def set_freshness(self, freshness: str) -> None:
        self._freshness = freshness

    def get_context(self) -> MacroContext:
        """Build aggregated macro context."""
        # Composite score: weighted average of signals
        cot_score = {"EXTREME_LONG": 1.0, "BULLISH": 0.5, "NEUTRAL": 0.0,
                     "BEARISH": -0.5, "EXTREME_SHORT": -1.0}.get(self._cot_bias, 0.0)

        macro_score = (
            0.4 * self._dxy_bias +
            0.3 * cot_score +
            0.3 * self._sentiment
        )

        return MacroContext(
            dxy_bias=self._dxy_bias,
            cot_bias=self._cot_bias,
            event_blocked=self._event_blocked,
            sentiment=self._sentiment,
            macro_score=round(macro_score, 3),
            freshness=self._freshness,
        )
