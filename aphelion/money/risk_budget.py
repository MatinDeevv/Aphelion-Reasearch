"""
APHELION Money Makers — Portfolio-Level Risk Budget

Tracks per-strategy risk consumption, enforces daily loss limits,
and manages the global risk budget across all active strategies.

Ensures no single strategy can consume more than its drawdown budget,
and provides real-time risk state for the SENTINEL and ARES layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from aphelion.core.config import SENTINEL

logger = logging.getLogger(__name__)


@dataclass
class StrategyRiskState:
    """Risk accounting state for a single strategy."""
    strategy_id: str
    daily_pnl: float = 0.0
    daily_loss_limit: float = 0.0           # Max daily loss (positive number = $ loss allowed)
    open_risk: float = 0.0                  # Current unrealised risk exposure
    trade_count_today: int = 0
    max_trades_per_day: int = 20
    is_halted: bool = False
    halt_reason: str = ""
    # Tracking
    total_pnl: float = 0.0
    peak_equity: float = 0.0
    current_drawdown: float = 0.0

    def update_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl
        self.total_pnl += pnl

    def new_day(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.trade_count_today = 0
        if not self.is_halted:
            self.halt_reason = ""


@dataclass
class RiskBudgetConfig:
    """Global risk budget configuration."""
    max_portfolio_daily_loss_pct: float = 0.03     # 3% of equity (L1)
    max_portfolio_drawdown_pct: float = 0.10       # 10% from peak (L3)
    max_correlation_exposure: float = 0.70          # Max allowed pair corr
    per_strategy_daily_loss_pct: float = 0.015      # 1.5% per strategy
    per_strategy_max_trades: int = 20               # Per day
    global_max_open_positions: int = 6              # Across all strategies
    risk_free_rate: float = 0.05                    # For real-time Sharpe


class RiskBudget:
    """
    Portfolio-level risk manager.
    Sits between Money Makers and SENTINEL, providing granular
    per-strategy risk accounting and global budget enforcement.
    """

    def __init__(
        self,
        initial_equity: float,
        config: Optional[RiskBudgetConfig] = None,
    ):
        self._config = config or RiskBudgetConfig()
        self._equity = initial_equity
        self._peak_equity = initial_equity
        self._daily_pnl: float = 0.0
        self._strategies: dict[str, StrategyRiskState] = {}
        self._global_open_positions: int = 0
        self._current_date: Optional[datetime] = None
        self._portfolio_halted: bool = False

    # ── Strategy Registration ────────────────────────────────────────────────

    def register_strategy(self, strategy_id: str) -> StrategyRiskState:
        """Register a strategy for risk tracking."""
        state = StrategyRiskState(
            strategy_id=strategy_id,
            daily_loss_limit=self._equity * self._config.per_strategy_daily_loss_pct,
            max_trades_per_day=self._config.per_strategy_max_trades,
        )
        self._strategies[strategy_id] = state
        return state

    # ── Pre-Trade Check ──────────────────────────────────────────────────────

    def can_trade(self, strategy_id: str, risk_amount: float = 0.0) -> tuple[bool, str]:
        """
        Check if a strategy is allowed to open a new trade.

        Returns:
            (allowed, reason) — reason is empty string if allowed.
        """
        if self._portfolio_halted:
            return False, "PORTFOLIO_HALTED"

        state = self._strategies.get(strategy_id)
        if state is None:
            return False, "STRATEGY_NOT_REGISTERED"

        if state.is_halted:
            return False, f"STRATEGY_HALTED: {state.halt_reason}"

        # Daily loss check (strategy-level)
        if abs(state.daily_pnl) >= state.daily_loss_limit and state.daily_pnl < 0:
            state.is_halted = True
            state.halt_reason = "DAILY_LOSS_LIMIT"
            return False, "DAILY_LOSS_LIMIT"

        # Trade count check
        if state.trade_count_today >= state.max_trades_per_day:
            return False, "MAX_TRADES_PER_DAY"

        # Global position limit
        if self._global_open_positions >= self._config.global_max_open_positions:
            return False, "GLOBAL_MAX_POSITIONS"

        # Portfolio daily loss check
        portfolio_daily_limit = self._equity * self._config.max_portfolio_daily_loss_pct
        if self._daily_pnl < -portfolio_daily_limit:
            self._portfolio_halted = True
            return False, "PORTFOLIO_DAILY_LOSS"

        # Portfolio drawdown check
        if self._peak_equity > 0:
            dd = 1 - self._equity / self._peak_equity
            if dd > self._config.max_portfolio_drawdown_pct:
                self._portfolio_halted = True
                return False, "PORTFOLIO_MAX_DRAWDOWN"

        return True, ""

    # ── Event Handlers ───────────────────────────────────────────────────────

    def on_trade_open(self, strategy_id: str, risk_amount: float) -> None:
        """Record a new trade opening."""
        state = self._strategies.get(strategy_id)
        if state:
            state.trade_count_today += 1
            state.open_risk += risk_amount
        self._global_open_positions += 1

    def on_trade_close(self, strategy_id: str, pnl: float) -> None:
        """Record a trade closing with its PnL."""
        state = self._strategies.get(strategy_id)
        if state:
            state.update_pnl(pnl)
            state.open_risk = max(0, state.open_risk - abs(pnl))
        self._daily_pnl += pnl
        self._global_open_positions = max(0, self._global_open_positions - 1)

    def update_equity(self, equity: float) -> None:
        """Update portfolio equity and peak tracking."""
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Update per-strategy loss limits based on new equity
        for state in self._strategies.values():
            state.daily_loss_limit = equity * self._config.per_strategy_daily_loss_pct

    def on_new_day(self, date: datetime) -> None:
        """Reset daily counters on new trading day."""
        if self._current_date is not None and date.date() == self._current_date.date():
            return
        self._current_date = date
        self._daily_pnl = 0.0
        self._portfolio_halted = False
        for state in self._strategies.values():
            state.new_day()
        logger.debug("Risk budget reset for new day: %s", date.date())

    # ── Analytics ────────────────────────────────────────────────────────────

    def get_risk_summary(self) -> dict:
        """Get a snapshot of the current risk state."""
        portfolio_dd = 0.0
        if self._peak_equity > 0:
            portfolio_dd = 1 - self._equity / self._peak_equity

        return {
            "equity": self._equity,
            "peak_equity": self._peak_equity,
            "portfolio_drawdown": portfolio_dd,
            "daily_pnl": self._daily_pnl,
            "global_open_positions": self._global_open_positions,
            "portfolio_halted": self._portfolio_halted,
            "strategies": {
                sid: {
                    "daily_pnl": s.daily_pnl,
                    "total_pnl": s.total_pnl,
                    "trade_count_today": s.trade_count_today,
                    "is_halted": s.is_halted,
                    "open_risk": s.open_risk,
                }
                for sid, s in self._strategies.items()
            },
        }

    @property
    def is_halted(self) -> bool:
        return self._portfolio_halted

    @property
    def strategies(self) -> dict[str, StrategyRiskState]:
        return dict(self._strategies)

    def reset(self) -> None:
        """Full reset."""
        self._daily_pnl = 0.0
        self._global_open_positions = 0
        self._portfolio_halted = False
        for s in self._strategies.values():
            s.new_day()
            s.is_halted = False
            s.total_pnl = 0.0
