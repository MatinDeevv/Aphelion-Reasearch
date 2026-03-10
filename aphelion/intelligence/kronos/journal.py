"""
KRONOS — Trade Journaling & Performance Analytics
Phase 13 — Engineering Spec v3.0

Tracks every trade with full context: ARES votes, features, regime,
session, outcome. Computes performance analytics for SOLA.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum

import numpy as np


@dataclass
class TradeRecord:
    """Full context record for a single trade."""
    trade_id: str
    timestamp_entry: datetime
    timestamp_exit: Optional[datetime] = None

    # Trade details
    direction: int = 0              # 1=BUY, -1=SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Outcome
    profit_usd: float = 0.0
    r_multiple: float = 0.0
    hold_time_seconds: float = 0.0
    outcome: str = ""               # "WIN", "LOSS", "BREAKEVEN"

    # Context at entry
    ares_confidence: float = 0.0
    ares_votes: Dict[str, dict] = field(default_factory=dict)
    regime: str = ""
    session: str = ""
    features_snapshot: Dict[str, float] = field(default_factory=dict)

    # Model performance
    hydra_confidence: float = 0.0
    hydra_direction: int = 0


class PerformanceMetrics:
    """Computed performance metrics from trade history."""

    def __init__(self, trades: List[TradeRecord]):
        self._trades = trades

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self._trades if t.profit_usd > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self._trades if t.profit_usd < 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_r_multiple(self) -> float:
        if not self._trades:
            return 0.0
        return float(np.mean([t.r_multiple for t in self._trades]))

    @property
    def total_profit(self) -> float:
        return sum(t.profit_usd for t in self._trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.profit_usd for t in self._trades if t.profit_usd > 0)
        gross_loss = abs(sum(t.profit_usd for t in self._trades if t.profit_usd < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    @property
    def max_drawdown(self) -> float:
        if not self._trades:
            return 0.0
        equity_curve = np.cumsum([t.profit_usd for t in self._trades])
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if len(self._trades) < 2:
            return 0.0
        returns = [t.profit_usd for t in self._trades]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        # Annualized: assume ~10 trades/day, 252 trading days
        return float(mean_ret / std_ret * np.sqrt(252 * 10))

    @property
    def consecutive_losses(self) -> int:
        max_streak = 0
        current = 0
        for t in self._trades:
            if t.profit_usd < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @property
    def rolling_win_rate_20(self) -> float:
        recent = self._trades[-20:]
        if not recent:
            return 0.0
        return sum(1 for t in recent if t.profit_usd > 0) / len(recent)

    @property
    def high_conf_win_rate(self) -> float:
        """Win rate for trades with ARES confidence > 0.7."""
        high_conf = [t for t in self._trades if t.ares_confidence > 0.7]
        if not high_conf:
            return 0.0
        return sum(1 for t in high_conf if t.profit_usd > 0) / len(high_conf)

    def win_rate_by_session(self) -> Dict[str, float]:
        by_session: Dict[str, List[TradeRecord]] = {}
        for t in self._trades:
            session = t.session or "UNKNOWN"
            by_session.setdefault(session, []).append(t)
        return {
            s: sum(1 for t in trades if t.profit_usd > 0) / len(trades)
            for s, trades in by_session.items()
            if trades
        }

    def win_rate_by_regime(self) -> Dict[str, float]:
        by_regime: Dict[str, List[TradeRecord]] = {}
        for t in self._trades:
            regime = t.regime or "UNKNOWN"
            by_regime.setdefault(regime, []).append(t)
        return {
            r: sum(1 for t in trades if t.profit_usd > 0) / len(trades)
            for r, trades in by_regime.items()
            if trades
        }

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_r_multiple": round(self.avg_r_multiple, 4),
            "total_profit": round(self.total_profit, 2),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "consecutive_losses": self.consecutive_losses,
        }


class KRONOSJournal:
    """
    Full trade journal with performance analytics.
    Sergeant-tier ARES awareness (2 votes).
    """

    def __init__(self):
        self._trades: List[TradeRecord] = []
        self._next_id = 1

    def record_entry(
        self,
        direction: int,
        entry_price: float,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        ares_confidence: float = 0.0,
        ares_votes: Optional[Dict[str, dict]] = None,
        regime: str = "",
        session: str = "",
        features: Optional[Dict[str, float]] = None,
        hydra_confidence: float = 0.0,
        hydra_direction: int = 0,
    ) -> str:
        """Record a new trade entry. Returns trade_id."""
        trade_id = f"T{self._next_id:06d}"
        self._next_id += 1

        record = TradeRecord(
            trade_id=trade_id,
            timestamp_entry=datetime.now(timezone.utc),
            direction=direction,
            entry_price=entry_price,
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ares_confidence=ares_confidence,
            ares_votes=ares_votes or {},
            regime=regime,
            session=session,
            features_snapshot=features or {},
            hydra_confidence=hydra_confidence,
            hydra_direction=hydra_direction,
        )
        self._trades.append(record)
        return trade_id

    def record_exit(
        self, trade_id: str, exit_price: float, profit_usd: float
    ) -> Optional[TradeRecord]:
        """Record trade exit with outcome."""
        for trade in self._trades:
            if trade.trade_id == trade_id:
                trade.timestamp_exit = datetime.now(timezone.utc)
                trade.exit_price = exit_price
                trade.profit_usd = profit_usd

                # Compute R-multiple
                risk = abs(trade.entry_price - trade.stop_loss)
                if risk > 0:
                    trade.r_multiple = (exit_price - trade.entry_price) * trade.direction / risk
                else:
                    trade.r_multiple = 0.0

                # Hold time
                trade.hold_time_seconds = (
                    trade.timestamp_exit - trade.timestamp_entry
                ).total_seconds()

                # Outcome
                if profit_usd > 0:
                    trade.outcome = "WIN"
                elif profit_usd < 0:
                    trade.outcome = "LOSS"
                else:
                    trade.outcome = "BREAKEVEN"

                return trade
        return None

    def get_metrics(self, last_n: Optional[int] = None) -> PerformanceMetrics:
        trades = self._trades[-last_n:] if last_n else self._trades
        return PerformanceMetrics(trades)

    @property
    def trades(self) -> List[TradeRecord]:
        return list(self._trades)

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        for t in self._trades:
            if t.trade_id == trade_id:
                return t
        return None
