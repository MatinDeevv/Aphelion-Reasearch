"""
HEPHAESTUS — Registry

Tracks all forged strategies — passed, failed, deployed, rejected.
Provides analytics over forge history.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aphelion.hephaestus.models import (
    ForgeResult,
    ForgeStatus,
    InputType,
    RejectionReport,
)

logger = logging.getLogger(__name__)


class HephaestusRegistry:
    """In-memory strategy registry and analytics store.

    Tracks every forge attempt so the system can learn which source types
    and indicator families tend to pass validation.
    """

    def __init__(self) -> None:
        self._results: dict[str, ForgeResult] = {}
        self._rejections: dict[str, RejectionReport] = {}

    # ── CRUD ─────────────────────────────────────────────────────────────

    def register(self, result: ForgeResult) -> None:
        """Store a forge result."""
        self._results[result.strategy_id] = result

    def register_rejection(self, report: RejectionReport) -> None:
        """Store a rejection report."""
        self._rejections[report.strategy_id] = report

    def get(self, strategy_id: str) -> Optional[ForgeResult]:
        """Look up a forge result by ID."""
        return self._results.get(strategy_id)

    def get_rejection(self, strategy_id: str) -> Optional[RejectionReport]:
        """Look up a rejection report by ID."""
        return self._rejections.get(strategy_id)

    # ── Queries ──────────────────────────────────────────────────────────

    def list_by_status(self, status: ForgeStatus) -> list[ForgeResult]:
        """All results with a given status."""
        return [r for r in self._results.values() if r.status == status]

    def list_deployed(self) -> list[ForgeResult]:
        """All currently deployed or shadowed strategies."""
        return [
            r
            for r in self._results.values()
            if r.status in (ForgeStatus.DEPLOYED, ForgeStatus.SHADOW)
        ]

    def list_rejected(self) -> list[ForgeResult]:
        """All rejected strategies."""
        return self.list_by_status(ForgeStatus.REJECTED)

    @property
    def total_forged(self) -> int:
        return len(self._results)

    @property
    def total_deployed(self) -> int:
        return len(self.list_deployed())

    @property
    def total_rejected(self) -> int:
        return len(self.list_rejected())

    # ── Analytics ────────────────────────────────────────────────────────

    def get_success_rate(self) -> float:
        """Overall success rate (deployed / total)."""
        if not self._results:
            return 0.0
        deployed = sum(
            1
            for r in self._results.values()
            if r.status in (ForgeStatus.DEPLOYED, ForgeStatus.SHADOW)
        )
        return deployed / len(self._results)

    def get_success_rate_by_source_type(self) -> dict[str, float]:
        """Success rate broken down by input type."""
        counts: dict[str, list[int]] = {}  # type → [total, deployed]
        for r in self._results.values():
            src = r.spec.source_type.value if r.spec else "UNKNOWN"
            if src not in counts:
                counts[src] = [0, 0]
            counts[src][0] += 1
            if r.status in (ForgeStatus.DEPLOYED, ForgeStatus.SHADOW):
                counts[src][1] += 1
        return {
            src: deployed / total if total > 0 else 0.0
            for src, (total, deployed) in counts.items()
        }

    def get_common_rejection_reasons(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Most common rejection reasons across all rejected strategies."""
        reason_counts: dict[str, int] = {}
        for r in self._results.values():
            if r.status == ForgeStatus.REJECTED and r.validation:
                for reason in r.validation.rejection_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_reasons[:top_n]

    def get_best_performing_deployed(self, top_n: int = 10) -> list[ForgeResult]:
        """Top N deployed strategies by validated Sharpe."""
        deployed = self.list_deployed()
        deployed.sort(
            key=lambda r: r.validation.sharpe_ratio if r.validation else 0.0,
            reverse=True,
        )
        return deployed[:top_n]

    def search(self, query: str) -> list[ForgeResult]:
        """Simple text search over strategy names and descriptions."""
        query_lower = query.lower()
        matches = []
        for r in self._results.values():
            if r.spec and (
                query_lower in r.spec.name.lower()
                or query_lower in r.spec.description.lower()
            ):
                matches.append(r)
        return matches
