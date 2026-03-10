"""
TITAN — Validation Reporter
Phase 15 — Engineering Spec v3.0

Generates detailed validation reports from TITAN gate results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from aphelion.risk.titan.gate import GateReport, GateStatus, ValidationResult

logger = logging.getLogger(__name__)


class TitanReporter:
    """Generates human-readable and JSON validation reports."""

    def __init__(self, report_dir: str = "reports/titan"):
        self._report_dir = report_dir
        self._reports: List[GateReport] = []

    def add_report(self, report: GateReport) -> None:
        self._reports.append(report)

    def generate_text(self, report: Optional[GateReport] = None) -> str:
        """Generate human-readable validation report."""
        r = report or (self._reports[-1] if self._reports else None)
        if r is None:
            return "No reports available."

        lines = [
            "=" * 60,
            "  TITAN Quality Gate Report",
            f"  Status: {r.status.value}",
            f"  Triggered by: {r.triggered_by}",
            f"  Timestamp: {r.timestamp.isoformat()}",
            f"  Duration: {r.duration_seconds:.1f}s",
            f"  Pass Rate: {r.pass_rate:.0%}",
            "=" * 60,
            "",
        ]

        # Validations
        for v in r.validations:
            icon = "PASS" if v.passed else "FAIL"
            lines.append(f"  [{icon}] {v.check_name}: "
                         f"{v.actual_value:.4f} (threshold: {v.threshold:.4f})")
            if v.message:
                lines.append(f"         {v.message}")

        if r.failures:
            lines.append("")
            lines.append("  FAILURES:")
            for f in r.failures:
                lines.append(f"    - {f}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_json(self, report: Optional[GateReport] = None) -> dict:
        """Generate JSON validation report."""
        r = report or (self._reports[-1] if self._reports else None)
        if r is None:
            return {}

        return {
            "status": r.status.value,
            "triggered_by": r.triggered_by,
            "timestamp": r.timestamp.isoformat(),
            "duration_seconds": r.duration_seconds,
            "pass_rate": r.pass_rate,
            "validations": [
                {
                    "check": v.check_name,
                    "passed": v.passed,
                    "actual": v.actual_value,
                    "threshold": v.threshold,
                    "message": v.message,
                }
                for v in r.validations
            ],
            "failures": r.failures,
        }

    @property
    def total_reports(self) -> int:
        return len(self._reports)

    @property
    def latest_status(self) -> Optional[GateStatus]:
        if self._reports:
            return self._reports[-1].status
        return None
