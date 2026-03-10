"""
ECHO — Pattern Library & Matching
Phase 13 — Engineering Spec v3.0

Stores winning trade setups as patterns. When a new bar occurs,
checks similarity to top-performing historical patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PatternFingerprint:
    """Encoded feature vector representing a trade setup."""
    pattern_id: str
    features: Dict[str, float]
    direction: int             # 1=BUY, -1=SELL
    outcome: str = ""          # "WIN", "LOSS"
    r_multiple: float = 0.0
    regime: str = ""
    session: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PatternEncoder:
    """Convert bar features into a normalized pattern fingerprint."""

    def __init__(self, key_features: Optional[List[str]] = None):
        self._key_features = key_features or [
            "rsi", "atr", "bb_percentile", "ema_cross",
            "adx", "macd_histogram", "volume_z", "stoch_k",
        ]

    def encode(self, features: Dict[str, float]) -> np.ndarray:
        """Extract key features into a fixed-length vector."""
        vector = []
        for key in self._key_features:
            vector.append(features.get(key, 0.0))
        return np.array(vector, dtype=np.float64)


class PatternLibrary:
    """
    Stores and matches trade setup patterns.
    Sergeant-tier ARES awareness (2 votes).
    """

    def __init__(self, max_patterns: int = 10000, similarity_threshold: float = 0.85):
        self._patterns: List[PatternFingerprint] = []
        self._encoder = PatternEncoder()
        self._max_patterns = max_patterns
        self._sim_threshold = similarity_threshold

    def store(self, fingerprint: PatternFingerprint) -> None:
        """Store a pattern."""
        self._patterns.append(fingerprint)
        if len(self._patterns) > self._max_patterns:
            # Remove oldest patterns
            self._patterns = self._patterns[-self._max_patterns:]

    def match(
        self, features: Dict[str, float], top_n: int = 5
    ) -> List[Tuple[PatternFingerprint, float]]:
        """Find most similar patterns to current features."""
        if not self._patterns:
            return []

        query_vec = self._encoder.encode(features)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        similarities = []
        for pattern in self._patterns:
            pat_vec = self._encoder.encode(pattern.features)
            pat_norm = np.linalg.norm(pat_vec)
            if pat_norm == 0:
                continue
            cosine_sim = float(np.dot(query_vec, pat_vec) / (query_norm * pat_norm))
            similarities.append((pattern, cosine_sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def get_confidence_boost(self, features: Dict[str, float]) -> float:
        """
        Get ARES confidence modifier based on pattern similarity.
        Positive = boost (similar to winning patterns).
        Negative = reduce (similar to losing patterns).
        """
        matches = self.match(features, top_n=10)
        if not matches:
            return 0.0

        winning_sim = 0.0
        losing_sim = 0.0
        for pattern, sim in matches:
            if sim < self._sim_threshold:
                continue
            if pattern.outcome == "WIN":
                winning_sim += sim * pattern.r_multiple
            elif pattern.outcome == "LOSS":
                losing_sim += sim * abs(pattern.r_multiple)

        total = winning_sim + losing_sim
        if total == 0:
            return 0.0

        # Normalized boost: positive if mostly winning, negative if mostly losing
        return (winning_sim - losing_sim) / total * 0.2  # Max ±0.2 boost

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def win_pattern_count(self) -> int:
        return sum(1 for p in self._patterns if p.outcome == "WIN")

    @property
    def loss_pattern_count(self) -> int:
        return sum(1 for p in self._patterns if p.outcome == "LOSS")

    def clear(self) -> None:
        self._patterns.clear()
