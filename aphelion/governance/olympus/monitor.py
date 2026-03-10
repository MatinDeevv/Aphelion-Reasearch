"""
OLYMPUS — Performance Monitor
Phase 20 — Engineering Spec v3.0

Real-time monitoring of strategy health, decay detection, and alerts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Health report for a strategy."""
    strategy: str
    is_healthy: bool
    win_rate: float = 0.0
    sharpe: float = 0.0
    drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    alert: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceMonitor:
    """
    Monitors ALPHA and OMEGA strategy health in real-time.
    Generates alerts when degradation thresholds are breached.
    """

    def __init__(
        self,
        min_win_rate: float = 0.48,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.08,
        max_consecutive_losses: int = 5,
    ):
        self._min_wr = min_win_rate
        self._min_sharpe = min_sharpe
        self._max_dd = max_drawdown
        self._max_cl = max_consecutive_losses
        self._reports: Dict[str, HealthReport] = {}

    def evaluate(
        self,
        strategy: str,
        win_rate: float,
        sharpe: float,
        drawdown_pct: float,
        consecutive_losses: int,
    ) -> HealthReport:
        """Evaluate strategy health and return report."""
        alerts = []
        healthy = True

        if win_rate < self._min_wr:
            alerts.append(f"WR {win_rate:.1%} < {self._min_wr:.1%}")
            healthy = False
        if sharpe < self._min_sharpe:
            alerts.append(f"Sharpe {sharpe:.2f} < {self._min_sharpe:.2f}")
            healthy = False
        if drawdown_pct > self._max_dd:
            alerts.append(f"DD {drawdown_pct:.1%} > {self._max_dd:.1%}")
            healthy = False
        if consecutive_losses >= self._max_cl:
            alerts.append(f"ConsecLoss {consecutive_losses} >= {self._max_cl}")
            healthy = False

        report = HealthReport(
            strategy=strategy,
            is_healthy=healthy,
            win_rate=win_rate,
            sharpe=sharpe,
            drawdown_pct=drawdown_pct,
            consecutive_losses=consecutive_losses,
            alert="; ".join(alerts) if alerts else "OK",
        )
        self._reports[strategy] = report
        return report

    def is_strategy_healthy(self, strategy: str) -> bool:
        rpt = self._reports.get(strategy)
        return rpt.is_healthy if rpt else True

    @property
    def all_healthy(self) -> bool:
        return all(r.is_healthy for r in self._reports.values())

    @property
    def reports(self) -> Dict[str, HealthReport]:
        return dict(self._reports)
