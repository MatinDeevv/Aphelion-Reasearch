"""
OLYMPUS — Reporter
Phase 20 — Engineering Spec v3.0

Generates operational reports for OLYMPUS orchestrator.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OlympusReporter:
    """
    Produces text and JSON reports on OLYMPUS state:
    allocation, strategy performance, decay status, alerts.
    """

    def generate_text_report(
        self,
        mode: str,
        system_state: str,
        alpha_pct: float,
        omega_pct: float,
        alpha_wr: float = 0.0,
        omega_wr: float = 0.0,
        alpha_sharpe: float = 0.0,
        omega_sharpe: float = 0.0,
        decay_detected: bool = False,
        retraining_needed: bool = False,
    ) -> str:
        lines = [
            "=" * 50,
            "  OLYMPUS STATUS REPORT",
            "=" * 50,
            f"  Mode:              {mode}",
            f"  System State:      {system_state}",
            f"  Allocation:        ALPHA {alpha_pct:.0%} / OMEGA {omega_pct:.0%}",
            "",
            f"  ALPHA  WR={alpha_wr:.1%}  Sharpe={alpha_sharpe:.2f}",
            f"  OMEGA  WR={omega_wr:.1%}  Sharpe={omega_sharpe:.2f}",
            "",
            f"  Decay Detected:    {'YES' if decay_detected else 'No'}",
            f"  Retraining Needed: {'YES' if retraining_needed else 'No'}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def generate_json_report(self, **kwargs: Any) -> str:
        payload = {
            "report": "OLYMPUS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        return json.dumps(payload, indent=2, default=str)
