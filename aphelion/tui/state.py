"""
APHELION TUI — Shared Mutable State  (v2 — Bloomberg-grade)

Single dataclass tree that every Textual widget reads from.
The PaperSession / event-bus writes into this object via TUIBridge.

New in v2
─────────
* Equity history deque for sparkline rendering (up to 500 ticks)
* Price tick history for mini-chart (up to 200 ticks)
* Performance metrics (Sharpe, profit-factor, max-DD, …)
* Alert queue for pop-up notifications
* HYDRA signal history for confidence sparkline
* Timestamp on every sub-view for "last updated" display
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Sub-views
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HydraSignalView:
    """Snapshot of last HYDRA prediction for the TUI."""
    direction: str = "FLAT"        # LONG / SHORT / FLAT
    confidence: float = 0.0
    uncertainty: float = 0.0
    probs_5m: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    probs_15m: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    probs_1h: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    horizon_agreement: float = 0.0
    gate_weights: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    moe_routing: list[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    top_features: list[tuple[str, float]] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    # v2 — history for confidence sparkline
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=120))
    signal_count: int = 0


@dataclass
class PositionView:
    """One open position shown in the TUI."""
    position_id: str = ""
    symbol: str = "XAUUSD"
    direction: str = "LONG"
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    size_lots: float = 0.0
    unrealized_pnl: float = 0.0
    open_time: Optional[datetime] = None
    # v2 — risk-to-reward ratio + hold duration
    risk_reward: float = 0.0
    hold_bars: int = 0


@dataclass
class SentinelView:
    """Current risk state snapshot for the TUI."""
    l1_triggered: bool = False
    l2_triggered: bool = False
    l3_triggered: bool = False
    open_positions: int = 0
    max_positions: int = 3
    total_exposure_pct: float = 0.0
    max_exposure_pct: float = 0.06
    daily_drawdown_pct: float = 0.0
    session_peak_equity: float = 0.0
    trading_allowed: bool = True
    circuit_breaker_active: bool = False
    # v2 — time-in-breaker tracking
    breaker_since: Optional[datetime] = None
    drawdown_history: deque = field(default_factory=lambda: deque(maxlen=120))


@dataclass
class EquityView:
    """Equity / PnL snapshot."""
    account_equity: float = 100_000.0
    session_peak: float = 100_000.0
    daily_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    # v2 — sparkline data and performance metrics
    equity_history: deque = field(default_factory=lambda: deque(maxlen=500))
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_hold_bars: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0


@dataclass
class PriceView:
    """Live price ticker data (v2)."""
    symbol: str = "XAUUSD"
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    high: float = 0.0
    low: float = 0.0
    spread: float = 0.0
    tick_history: deque = field(default_factory=lambda: deque(maxlen=200))


@dataclass
class AlertEntry:
    """Pop-up / banner alert (v2)."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "INFO"   # INFO, WARNING, CRITICAL
    title: str = ""
    message: str = ""
    acknowledged: bool = False


@dataclass
class LogEntry:
    """Single event log entry."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: str = "INFO"  # INFO, WARNING, ERROR, FILL, REJECT, SENTINEL, HYDRA
    message: str = ""


@dataclass
class TUIState:
    """
    Shared mutable state for the APHELION TUI (v2 — Bloomberg-grade).

    The paper-session loop or event-bus callbacks write here;
    the Textual widgets read here every refresh cycle.
    """
    # Session
    session_name: str = "Paper-01"
    session_start: Optional[datetime] = None
    market_open: bool = False
    current_session: str = "DEAD_ZONE"
    current_time: Optional[datetime] = None

    # HYDRA
    hydra: HydraSignalView = field(default_factory=HydraSignalView)

    # SENTINEL
    sentinel: SentinelView = field(default_factory=SentinelView)

    # Equity
    equity: EquityView = field(default_factory=EquityView)

    # Price ticker (v2)
    price: PriceView = field(default_factory=PriceView)

    # Positions
    positions: list[PositionView] = field(default_factory=list)

    # Log
    log: list[LogEntry] = field(default_factory=list)
    max_log_lines: int = 100

    # Alerts (v2)
    alerts: list[AlertEntry] = field(default_factory=list)
    max_alerts: int = 50

    # Bar counter
    bars_processed: int = 0
    last_bar_time: Optional[datetime] = None

    # System stats (v2)
    cpu_usage: float = 0.0
    memory_mb: float = 0.0
    latency_ms: float = 0.0
    uptime_seconds: float = 0.0

    # Feed status (v3 — Phase 5)
    feed_connected: bool = False
    feed_mode: str = "SIMULATED"
    feed_ticks_per_min: float = 0.0
    feed_reconnect_count: int = 0
    session_duration_min: float = 0.0
    sentinel_rejections: int = 0

    def push_log(self, level: str, message: str) -> None:
        """Append a log entry, trimming old entries if needed."""
        self.log.append(LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
        ))
        if len(self.log) > self.max_log_lines:
            self.log = self.log[-self.max_log_lines:]

    def push_alert(self, severity: str, title: str, message: str) -> None:
        """Push a new alert notification."""
        self.alerts.append(AlertEntry(
            severity=severity,
            title=title,
            message=message,
        ))
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

    def push_equity_tick(self, equity: float) -> None:
        """Record equity point for sparkline."""
        self.equity.equity_history.append(equity)
        # Update max drawdown
        if self.equity.session_peak > 0:
            dd = (self.equity.session_peak - equity) / self.equity.session_peak
            self.equity.max_drawdown_pct = max(self.equity.max_drawdown_pct, dd)

    def push_price_tick(self, bid: float, ask: float) -> None:
        """Record price tick for mini-chart."""
        mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0
        self.price.bid = bid
        self.price.ask = ask
        self.price.last = mid
        self.price.spread = ask - bid
        self.price.tick_history.append(mid)

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction [0, 1]."""
        total = self.equity.total_trades
        return self.equity.winning_trades / total if total > 0 else 0.0

    @property
    def unacknowledged_alerts(self) -> list[AlertEntry]:
        """Return alerts not yet acknowledged."""
        return [a for a in self.alerts if not a.acknowledged]
