"""
APHELION Feature Registry
Tracks feature metadata, versioning, importance scores, and redundancy detection.
Phase 1 v2 — Engineering Spec v3.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FeatureRecord:
    """Metadata for a single registered feature."""
    name: str
    version: str
    dtype: str                    # "continuous", "categorical", "binary"
    normalizer: str               # "zscore", "minmax", "none"
    active: bool = True
    importance_score: float = 0.0  # Updated by SOLA
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""
    source_module: str = ""


class FeatureRegistry:
    """
    Central registry for all APHELION features.
    Tracks metadata, importance, and cross-feature correlation.
    """

    def __init__(self):
        self._features: Dict[str, FeatureRecord] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._feature_names_ordered: List[str] = []

    def register(self, record: FeatureRecord) -> None:
        """Register or update a feature."""
        self._features[record.name] = record

    def deactivate(self, name: str) -> None:
        """Deactivate a feature (still tracked, not used in models)."""
        if name in self._features:
            self._features[name].active = False

    def activate(self, name: str) -> None:
        if name in self._features:
            self._features[name].active = True

    def is_active(self, name: str) -> bool:
        return self._features.get(name, FeatureRecord(name="", version="0", dtype="", normalizer="")).active

    def get(self, name: str) -> Optional[FeatureRecord]:
        return self._features.get(name)

    def list_active(self) -> List[FeatureRecord]:
        return [f for f in self._features.values() if f.active]

    def list_all(self) -> List[FeatureRecord]:
        return list(self._features.values())

    @property
    def active_count(self) -> int:
        return sum(1 for f in self._features.values() if f.active)

    @property
    def total_count(self) -> int:
        return len(self._features)

    def update_importance(self, name: str, score: float) -> None:
        """Update feature importance score (called by SOLA)."""
        if name in self._features:
            self._features[name].importance_score = score
            self._features[name].last_updated = datetime.now(timezone.utc)

    def update_correlation_matrix(self, feature_data: Dict[str, np.ndarray]) -> None:
        """Compute pairwise correlation matrix from recent feature data."""
        names = sorted(feature_data.keys())
        if len(names) < 2:
            return

        n = len(names)
        matrix = np.zeros((n, n))
        arrays = [feature_data[name] for name in names]

        for i in range(n):
            for j in range(i, n):
                if len(arrays[i]) == len(arrays[j]) and len(arrays[i]) > 1:
                    std_i = np.std(arrays[i])
                    std_j = np.std(arrays[j])
                    if std_i > 0 and std_j > 0:
                        corr = float(np.corrcoef(arrays[i], arrays[j])[0, 1])
                        if np.isnan(corr):
                            corr = 0.0
                    else:
                        corr = 0.0
                else:
                    corr = 0.0
                matrix[i, j] = corr
                matrix[j, i] = corr

        self._correlation_matrix = matrix
        self._feature_names_ordered = names

    def get_redundant_pairs(self, threshold: float = 0.95) -> List[tuple]:
        """Find feature pairs with correlation above threshold."""
        if self._correlation_matrix is None:
            return []

        pairs = []
        n = len(self._feature_names_ordered)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self._correlation_matrix[i, j]) > threshold:
                    pairs.append((
                        self._feature_names_ordered[i],
                        self._feature_names_ordered[j],
                        float(self._correlation_matrix[i, j]),
                    ))
        return pairs

    def get_top_features(self, n: int = 10) -> List[FeatureRecord]:
        """Get top N features by importance score."""
        active = self.list_active()
        return sorted(active, key=lambda f: f.importance_score, reverse=True)[:n]

    def to_dict(self) -> Dict[str, dict]:
        """Serialize registry."""
        return {
            name: {
                "version": rec.version,
                "dtype": rec.dtype,
                "normalizer": rec.normalizer,
                "active": rec.active,
                "importance_score": rec.importance_score,
                "source_module": rec.source_module,
            }
            for name, rec in self._features.items()
        }


# Singleton registry
_GLOBAL_REGISTRY: Optional[FeatureRegistry] = None


def get_registry() -> FeatureRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = FeatureRegistry()
    return _GLOBAL_REGISTRY
