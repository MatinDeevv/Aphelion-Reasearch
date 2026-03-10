"""
ECHO — Pattern Matcher
Phase 13 — Engineering Spec v3.0

Real-time pattern matching against the PatternLibrary.
Computes similarity scores and generates match signals for ARES.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from aphelion.intelligence.echo.library import (
    PatternEncoder,
    PatternFingerprint,
    PatternLibrary,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching current features against the pattern library."""
    matched_pattern_id: str
    similarity: float
    direction: int
    confidence: float
    avg_r_multiple: float
    win_rate: float


class PatternMatcher:
    """
    Real-time pattern matching engine.
    On each bar, encodes current features and finds top-K nearest neighbours
    from the library. Generates an aggregated match signal for ARES.
    """

    def __init__(
        self,
        library: Optional[PatternLibrary] = None,
        encoder: Optional[PatternEncoder] = None,
        top_k: int = 5,
        min_similarity: float = 0.70,
    ):
        self._library = library or PatternLibrary()
        self._encoder = encoder or PatternEncoder()
        self._top_k = top_k
        self._min_similarity = min_similarity

    def match(self, current_features: Dict[str, float]) -> List[MatchResult]:
        """Find top-K matching patterns for the current feature set."""
        current_vec = self._encoder.encode(current_features)
        matches = self._library.find_similar(current_vec, top_k=self._top_k)

        results = []
        for pattern, similarity in matches:
            if similarity < self._min_similarity:
                continue
            results.append(MatchResult(
                matched_pattern_id=pattern.pattern_id,
                similarity=similarity,
                direction=pattern.direction,
                confidence=similarity * 0.8,
                avg_r_multiple=pattern.r_multiple,
                win_rate=1.0 if pattern.outcome == "WIN" else 0.0,
            ))
        return results

    def generate_signal(self, current_features: Dict[str, float]) -> Tuple[int, float]:
        """Generate direction + confidence from pattern matching.

        Returns:
            (direction, confidence) where direction is -1, 0, or 1.
        """
        matches = self.match(current_features)
        if not matches:
            return 0, 0.0

        # Majority vote weighted by similarity
        weighted_dir = sum(m.direction * m.similarity for m in matches)
        total_sim = sum(m.similarity for m in matches)

        if total_sim < 1e-10:
            return 0, 0.0

        score = weighted_dir / total_sim
        if score > 0.3:
            direction = 1
        elif score < -0.3:
            direction = -1
        else:
            direction = 0

        confidence = min(1.0, abs(score))
        avg_wr = np.mean([m.win_rate for m in matches])
        confidence *= avg_wr  # Discount by historical win rate

        return direction, float(confidence)
