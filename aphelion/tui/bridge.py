"""
APHELION TUI — Paper Session Bridge  (v2 — Bloomberg-grade)

Connects the PaperSession event loop to TUI state for live dashboard updates.
Now includes:
  • Rate-throttled updates (configurable min interval)
  • Equity history tracking for sparklines
  • Price tick recording
  • Confidence history for HYDRA sparkline
  • Drawdown history for risk sparkline
  • Alert generation on circuit breaker trips
  • System stats updates (CPU, memory, latency)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from aphelion.tui.state import (
    TUIState,
    HydraSignalView,
    SentinelView,
    EquityView,
    PositionView,
    PriceView,
)

logger = logging.getLogger(__name__)


class TUIBridge:
    """
    Bridges between the PaperSession / event system and TUI state.

    v2 additions: throttling, history tracking, alert generation.
    """

    def __init__(
        self,
        state: TUIState,
        min_update_interval: float = 0.1,
    ):
        self._state = state
        self._min_interval = min_update_interval
        self._last_update: dict[str, float] = {}

    @property
    def state(self) -> TUIState:
        return self._state

    def _should_update(self, channel: str) -> bool:
        """Rate-gate: skip if last update was < min_interval ago."""
        now = time.monotonic()
        last = self._last_update.get(channel, 0.0)
        if now - last < self._min_interval:
            return False
        self._last_update[channel] = now
        return True

    # ── Bar Tick ────────────────────────────────────────────────

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

    # ── HYDRA Signal ────────────────────────────────────────────

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
        # v2: track confidence history for sparkline
        h.confidence_history.append(confidence)
        h.signal_count += 1

    # ── SENTINEL Risk ───────────────────────────────────────────

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
        was_active = s.circuit_breaker_active
        s.l1_triggered = l1
        s.l2_triggered = l2
        s.l3_triggered = l3
        s.open_positions = open_positions
        s.total_exposure_pct = total_exposure_pct
        s.daily_drawdown_pct = daily_drawdown_pct
        s.session_peak_equity = session_peak_equity
        s.trading_allowed = trading_allowed
        s.circuit_breaker_active = l1 or l2 or l3

        # v2: track breaker activation time
        if s.circuit_breaker_active and not was_active:
            s.breaker_since = datetime.now(timezone.utc)
            self._state.push_alert(
                "CRITICAL",
                "Circuit Breaker Activated",
                f"L1={l1} L2={l2} L3={l3} DD={daily_drawdown_pct*100:.2f}%",
            )
        elif not s.circuit_breaker_active and was_active:
            s.breaker_since = None
            self._state.push_alert(
                "INFO",
                "Circuit Breaker Cleared",
                "All breaker levels reset",
            )

        # v2: drawdown history for sparkline
        s.drawdown_history.append(daily_drawdown_pct)

    # ── Equity ──────────────────────────────────────────────────

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
        # v2: equity history for sparkline
        self._state.push_equity_tick(account_equity)

    # ── Positions ───────────────────────────────────────────────

    def update_positions(self, positions: list[PositionView]) -> None:
        """Replace the open positions list."""
        self._state.positions = positions

    # ── Price Ticker (v2) ───────────────────────────────────────

    def update_price(
        self,
        bid: float,
        ask: float,
        change: float = 0.0,
        change_pct: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
    ) -> None:
        """Push live price data into TUI."""
        if not self._should_update("price"):
            return
        p = self._state.price
        p.bid = bid
        p.ask = ask
        p.change = change
        p.change_pct = change_pct
        p.high = high
        p.low = low
        self._state.push_price_tick(bid, ask)

    # ── Performance Metrics (v2) ────────────────────────────────

    def update_performance(
        self,
        sharpe_ratio: float = 0.0,
        profit_factor: float = 0.0,
        max_drawdown_pct: float = 0.0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        best_trade: float = 0.0,
        worst_trade: float = 0.0,
        avg_hold_bars: float = 0.0,
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
    ) -> None:
        """Push performance metrics into TUI equity view."""
        e = self._state.equity
        e.sharpe_ratio = sharpe_ratio
        e.profit_factor = profit_factor
        e.max_drawdown_pct = max_drawdown_pct
        e.avg_win = avg_win
        e.avg_loss = avg_loss
        e.best_trade = best_trade
        e.worst_trade = worst_trade
        e.avg_hold_bars = avg_hold_bars
        e.consecutive_wins = consecutive_wins
        e.consecutive_losses = consecutive_losses

    # ── System Stats (v2) ───────────────────────────────────────

    def update_system_stats(
        self,
        cpu_usage: float = 0.0,
        memory_mb: float = 0.0,
        latency_ms: float = 0.0,
        uptime_seconds: float = 0.0,
    ) -> None:
        """Push system health metrics."""
        if not self._should_update("system"):
            return
        self._state.cpu_usage = cpu_usage
        self._state.memory_mb = memory_mb
        self._state.latency_ms = latency_ms
        self._state.uptime_seconds = uptime_seconds

    # ── Logging ─────────────────────────────────────────────────

    def log(self, level: str, message: str) -> None:
        """Push a log entry into the TUI event log."""
        self._state.push_log(level, message)

    def alert(self, severity: str, title: str, message: str) -> None:
        """Push an alert notification."""
        self._state.push_alert(severity, title, message)

    # ── Feed Status (v3 — Phase 5) ─────────────────────────────

    def update_feed_status(
        self,
        connected: bool,
        mode: str = "LIVE",
        ticks_per_min: float = 0.0,
        reconnect_count: int = 0,
    ) -> None:
        """Push feed connection and throughput stats."""
        self._state.feed_connected = connected
        self._state.feed_mode = mode
        self._state.feed_ticks_per_min = ticks_per_min
        self._state.feed_reconnect_count = reconnect_count

    def update_session_duration(self, duration_minutes: float) -> None:
        """Push session elapsed time."""
        self._state.session_duration_min = duration_minutes

    def update_sentinel_rejections(self, count: int) -> None:
        """Push sentinel rejection counter."""
        self._state.sentinel_rejections = count
