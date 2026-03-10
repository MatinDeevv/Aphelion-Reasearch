"""
HERALD — News/Event Processor Core
Phase 19 — Engineering Spec v3.0

Processes economic news events and classifies their impact on gold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """A classified news event."""
    headline: str
    impact: str          # HIGH, MEDIUM, LOW
    gold_bias: int       # -1, 0, 1
    confidence: float    # [0, 1]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HeraldCore:
    """
    Classifies economic events by impact on gold.
    Maintains an event queue for ARES to consume.
    """

    GOLD_POSITIVE: List[str] = [
        "rate cut", "dovish", "inflation rise", "geopolitical",
        "safe haven", "debt ceiling", "recession", "QE",
    ]
    GOLD_NEGATIVE: List[str] = [
        "rate hike", "hawkish", "inflation fall", "strong USD",
        "taper", "risk-on", "growth surprise",
    ]

    def __init__(self, max_queue: int = 100):
        self._queue: List[NewsEvent] = []
        self._max_queue = max_queue

    def classify(self, headline: str, impact: str = "MEDIUM") -> NewsEvent:
        """Classify a news headline for gold impact."""
        lower = headline.lower()
        bias = 0
        conf = 0.3

        for keyword in self.GOLD_POSITIVE:
            if keyword in lower:
                bias = 1
                conf = 0.6
                break

        if bias == 0:
            for keyword in self.GOLD_NEGATIVE:
                if keyword in lower:
                    bias = -1
                    conf = 0.6
                    break

        if impact == "HIGH":
            conf = min(1.0, conf + 0.2)

        event = NewsEvent(headline=headline, impact=impact, gold_bias=bias, confidence=conf)
        self._queue.append(event)
        if len(self._queue) > self._max_queue:
            self._queue = self._queue[-self._max_queue:]
        return event

    @property
    def recent_events(self) -> List[NewsEvent]:
        return self._queue[-20:]

    @property
    def net_bias(self) -> float:
        """Weighted average gold bias from recent events."""
        if not self._queue:
            return 0.0
        recent = self._queue[-10:]
        return sum(e.gold_bias * e.confidence for e in recent) / len(recent)
