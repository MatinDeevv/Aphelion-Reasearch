"""
APHELION TUI — Shared Mutable State
Single dataclass that all screens read from.
The paper-session / event-bus writes into this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


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


@dataclass
class LogEntry:
    """Single event log entry."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    level: str = "INFO"  # INFO, WARNING, ERROR, FILL, REJECT
    message: str = ""


@dataclass
class TUIState:
    """
    Shared mutable state for the APHELION TUI.

    The paper-session loop or event-bus callbacks write here;
    the dashboard screens read here every refresh cycle.
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

    # Positions
    positions: list[PositionView] = field(default_factory=list)

    # Log
    log: list[LogEntry] = field(default_factory=list)
    max_log_lines: int = 100

    # Bar counter
    bars_processed: int = 0
    last_bar_time: Optional[datetime] = None

    def push_log(self, level: str, message: str) -> None:
        """Append a log entry, trimming old entries if needed."""
        self.log.append(LogEntry(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
        ))
        if len(self.log) > self.max_log_lines:
            self.log = self.log[-self.max_log_lines:]
