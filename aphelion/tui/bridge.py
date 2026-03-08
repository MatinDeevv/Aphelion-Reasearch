"""
APHELION TUI — Paper Session Bridge
Connects the PaperSession event loop to TUI state for live dashboard updates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aphelion.tui.state import (
    TUIState,
    HydraSignalView,
    SentinelView,
    EquityView,
    PositionView,
)

logger = logging.getLogger(__name__)


class TUIBridge:
    """
    Bridges between the PaperSession / event system and the TUI state.

    Call update_* methods from the paper-session loop to push data
    into the shared TUIState that the dashboard reads.
    """

    def __init__(self, state: TUIState):
        self._state = state

    @property
    def state(self) -> TUIState:
        return self._state

    def update_bar(
        self,
        bar_time: datetime,
        session_name: str,
        market_open: bool,
        bars_processed: int,
    ) -> None:
        """Called each bar tick."""
        self._state.current_time = bar_time
        self._state.current_session = session_name
        self._state.market_open = market_open
        self._state.bars_processed = bars_processed
        self._state.last_bar_time = bar_time

    def update_hydra_signal(
        self,
        direction: str,
        confidence: float,
        uncertainty: float,
        probs_5m: list[float],
        probs_15m: list[float],
        probs_1h: list[float],
        horizon_agreement: float,
        gate_weights: list[float],
        moe_routing: list[float],
        top_features: Optional[list[tuple[str, float]]] = None,
    ) -> None:
        """Push a new HYDRA signal into TUI state."""
        h = self._state.hydra
        h.direction = direction
        h.confidence = confidence
        h.uncertainty = uncertainty
        h.probs_5m = probs_5m
        h.probs_15m = probs_15m
        h.probs_1h = probs_1h
        h.horizon_agreement = horizon_agreement
        h.gate_weights = gate_weights
        h.moe_routing = moe_routing
        h.top_features = top_features or []
        h.timestamp = datetime.now(timezone.utc)

    def update_sentinel(
        self,
        l1: bool,
        l2: bool,
        l3: bool,
        open_positions: int,
        total_exposure_pct: float,
        daily_drawdown_pct: float,
        session_peak_equity: float,
        trading_allowed: bool,
    ) -> None:
        """Push SENTINEL risk state into TUI."""
        s = self._state.sentinel
        s.l1_triggered = l1
        s.l2_triggered = l2
        s.l3_triggered = l3
        s.open_positions = open_positions
        s.total_exposure_pct = total_exposure_pct
        s.daily_drawdown_pct = daily_drawdown_pct
        s.session_peak_equity = session_peak_equity
        s.trading_allowed = trading_allowed
        s.circuit_breaker_active = l1 or l2 or l3

    def update_equity(
        self,
        account_equity: float,
        daily_pnl: float,
        realized_pnl: float,
        unrealized_pnl: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
    ) -> None:
        """Push equity snapshot into TUI."""
        e = self._state.equity
        e.account_equity = account_equity
        e.session_peak = max(e.session_peak, account_equity)
        e.daily_pnl = daily_pnl
        e.realized_pnl = realized_pnl
        e.unrealized_pnl = unrealized_pnl
        e.total_trades = total_trades
        e.winning_trades = winning_trades
        e.losing_trades = losing_trades

    def update_positions(self, positions: list[PositionView]) -> None:
        """Replace the open positions list."""
        self._state.positions = positions

    def log(self, level: str, message: str) -> None:
        """Push a log entry into the TUI event log."""
        self._state.push_log(level, message)
