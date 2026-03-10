"""
APHELION KRONOS — Report Generator
Phase 13 — Engineering Spec v3.0

Generates human-readable and machine-parseable performance reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from aphelion.intelligence.kronos.analytics import KronosAnalytics, PerformanceSnapshot

logger = logging.getLogger(__name__)


class KronosReportGenerator:
    """Generates structured performance reports from analytics data."""

    def __init__(self, analytics: KronosAnalytics):
        self._analytics = analytics

    def generate_text_report(self) -> str:
        """Generate human-readable performance report."""
        snap = self._analytics.compute_snapshot()
        lines = [
            "=" * 60,
            "  APHELION KRONOS Performance Report",
            f"  Generated: {snap.timestamp.isoformat()}",
            "=" * 60,
            "",
            f"  Total Trades:     {snap.total_trades}",
            f"  Win Rate:         {snap.win_rate:.1%}",
            f"  Sharpe Ratio:     {snap.sharpe_ratio:.2f}",
            f"  Sortino Ratio:    {snap.sortino_ratio:.2f}",
            f"  Profit Factor:    {snap.profit_factor:.2f}",
            f"  Max Drawdown:     {snap.max_drawdown_pct:.1%}",
            f"  Avg Win:          ${snap.avg_win:.2f}",
            f"  Avg Loss:         ${snap.avg_loss:.2f}",
            f"  Expectancy:       ${snap.expectancy:.2f}",
            f"  Avg Duration:     {snap.avg_trade_duration_minutes:.1f} min",
            "",
        ]

        # Regime breakdown
        regime_stats = self._analytics.compute_by_regime()
        if regime_stats:
            lines.append("  --- By Regime ---")
            for regime, stats in regime_stats.items():
                lines.append(f"    {regime}: WR={stats.win_rate:.0%} "
                             f"Sharpe={stats.sharpe_ratio:.2f} "
                             f"PF={stats.profit_factor:.2f} "
                             f"({stats.total_trades} trades)")
            lines.append("")

        # Session breakdown
        session_stats = self._analytics.compute_by_session()
        if session_stats:
            lines.append("  --- By Session ---")
            for session, stats in session_stats.items():
                lines.append(f"    {session}: WR={stats.win_rate:.0%} "
                             f"Sharpe={stats.sharpe_ratio:.2f} "
                             f"PF={stats.profit_factor:.2f} "
                             f"({stats.total_trades} trades)")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_json_report(self) -> dict:
        """Generate machine-parseable JSON report."""
        snap = self._analytics.compute_snapshot()
        return {
            "timestamp": snap.timestamp.isoformat(),
            "overall": {
                "total_trades": snap.total_trades,
                "win_rate": snap.win_rate,
                "sharpe_ratio": snap.sharpe_ratio,
                "sortino_ratio": snap.sortino_ratio,
                "profit_factor": snap.profit_factor,
                "max_drawdown_pct": snap.max_drawdown_pct,
                "avg_win": snap.avg_win,
                "avg_loss": snap.avg_loss,
                "expectancy": snap.expectancy,
                "avg_trade_duration_min": snap.avg_trade_duration_minutes,
            },
            "by_regime": {
                r: {
                    "win_rate": s.win_rate,
                    "sharpe_ratio": s.sharpe_ratio,
                    "profit_factor": s.profit_factor,
                    "total_trades": s.total_trades,
                }
                for r, s in self._analytics.compute_by_regime().items()
            },
            "by_session": {
                s: {
                    "win_rate": st.win_rate,
                    "sharpe_ratio": st.sharpe_ratio,
                    "profit_factor": st.profit_factor,
                    "total_trades": st.total_trades,
                }
                for s, st in self._analytics.compute_by_session().items()
            },
        }
