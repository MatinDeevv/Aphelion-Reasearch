"""
APHELION KRONOS — Performance Analytics
Phase 13 — Engineering Spec v3.0

Computes rolling performance statistics, regime-aware analysis,
and generates data for the report generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Completed trade record for analytics."""
    trade_id: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str = ""
    regime: str = ""
    session: str = ""


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics."""
    timestamp: datetime
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float
    expectancy: float
    avg_trade_duration_minutes: float


class KronosAnalytics:
    """Rolling performance analytics engine for KRONOS journal entries."""

    def __init__(self, rolling_window: int = 200):
        self._trades: List[TradeRecord] = []
        self._rolling_window = rolling_window
        self._equity_curve: List[float] = [0.0]

    def add_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)
        self._equity_curve.append(self._equity_curve[-1] + trade.pnl)

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    def compute_snapshot(self) -> PerformanceSnapshot:
        """Compute current performance snapshot."""
        recent = self._trades[-self._rolling_window:]
        if not recent:
            return PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_trades=0, win_rate=0.0, sharpe_ratio=0.0,
                sortino_ratio=0.0, profit_factor=0.0, max_drawdown_pct=0.0,
                avg_win=0.0, avg_loss=0.0, expectancy=0.0,
                avg_trade_duration_minutes=0.0,
            )

        pnls = np.array([t.pnl for t in recent])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Sharpe
        mean_ret = float(np.mean(pnls))
        std_ret = float(np.std(pnls))
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Sortino
        downside = pnls[pnls < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-10
        sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Profit factor
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown
        eq = np.array(self._equity_curve[-self._rolling_window:])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        # Avg trade duration
        durations = []
        for t in recent:
            dt = (t.exit_time - t.entry_time).total_seconds() / 60.0
            durations.append(dt)
        avg_dur = float(np.mean(durations)) if durations else 0.0

        return PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_trades=len(recent),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            avg_trade_duration_minutes=avg_dur,
        )

    def compute_by_regime(self) -> Dict[str, PerformanceSnapshot]:
        """Compute performance breakdown by regime."""
        regimes: Dict[str, List[TradeRecord]] = {}
        for t in self._trades:
            regime = t.regime or "UNKNOWN"
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(t)

        results = {}
        for regime, trades in regimes.items():
            sub = KronosAnalytics(self._rolling_window)
            for t in trades:
                sub.add_trade(t)
            results[regime] = sub.compute_snapshot()
        return results

    def compute_by_session(self) -> Dict[str, PerformanceSnapshot]:
        """Compute performance breakdown by session."""
        sessions: Dict[str, List[TradeRecord]] = {}
        for t in self._trades:
            session = t.session or "UNKNOWN"
            if session not in sessions:
                sessions[session] = []
            sessions[session].append(t)

        results = {}
        for session, trades in sessions.items():
            sub = KronosAnalytics(self._rolling_window)
            for t in trades:
                sub.add_trade(t)
            results[session] = sub.compute_snapshot()
        return results
