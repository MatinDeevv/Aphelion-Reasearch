"""SENTINEL core risk authority and hard-limit state tracking.

Improvements:
- L1 (warning) and L2 (halt) drawdown triggers now functional
- P&L calculation uses lot_size_oz multiplier (1 lot = 100 oz for XAU/USD)
- Daily reset mechanism for new trading day
- is_trading_allowed checks L2 halt condition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date

from aphelion.core.clock import MarketClock
from aphelion.core.config import EventTopic, SENTINEL
from aphelion.core.event_bus import Event, EventBus, Priority
from aphelion.risk.sentinel.sentinel_v2 import (
    SentinelV2,
    CorrelationGuard,
    LatencyMonitor,
    CascadeProtection,
    DynamicSizer,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Active position tracked by SENTINEL."""

    position_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    size_lots: float
    size_pct: float  # Fraction of account (0.02 = 2%)
    open_time: datetime
    unrealized_pnl: float = 0.0


class SentinelCore:
    """Supreme, immutable risk authority for live trading controls.

    Three-tier drawdown protection:
    - L1 (3%): Warning — reduces position sizing by 50%
    - L2 (6%): Halt — no new trades, close-only mode
    - L3 (10%): Disconnect — emergency shutdown
    """

    def __init__(self, event_bus: EventBus, clock: MarketClock):
        self._event_bus = event_bus
        self._clock = clock
        self._positions: dict[str, Position] = {}
        self._account_equity: float = 0.0
        self._session_peak_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._trade_count_today: int = 0
        self._l3_triggered: bool = False
        self._l2_triggered: bool = False
        self._l1_triggered: bool = False
        self._current_day: date | None = None
        self._event_bus.subscribe(EventTopic.RISK, self._on_risk_event)

        # v2: Advanced risk sub-systems
        self._v2 = SentinelV2()
        self._regime: str = "UNKNOWN"
        self._current_atr: float = 0.0
        self._avg_atr: float = 0.0

    @property
    def l1_triggered(self) -> bool:
        return self._l1_triggered

    @property
    def l2_triggered(self) -> bool:
        return self._l2_triggered

    @property
    def l3_triggered(self) -> bool:
        return self._l3_triggered

    def daily_reset(self) -> None:
        """Reset daily counters for a new trading day."""
        today = self._clock.now_utc().date()
        if self._current_day is not None and self._current_day == today:
            return  # Already reset today
        self._current_day = today
        self._daily_pnl = 0.0
        self._trade_count_today = 0
        self._session_peak_equity = self._account_equity
        self._l1_triggered = False
        self._l2_triggered = False
        # L3 does NOT reset — requires manual intervention
        logger.info("SENTINEL daily reset for %s", today)

    def update_equity(self, equity: float) -> None:
        self._account_equity = equity

        # Check for new trading day
        now = self._clock.now_utc()
        if self._current_day is None or now.date() != self._current_day:
            self.daily_reset()

        if equity > self._session_peak_equity:
            self._session_peak_equity = equity

        drawdown = 0.0
        if self._session_peak_equity > 0:
            drawdown = (self._session_peak_equity - equity) / self._session_peak_equity

        # L1: Warning — reduce sizing
        if drawdown >= SENTINEL.daily_equity_drawdown_l1 and not self._l1_triggered:
            self._l1_triggered = True
            logger.warning("SENTINEL L1 WARNING: %.1f%% drawdown — reducing position sizes", drawdown * 100)
            self._event_bus.publish_nowait(
                Event(
                    topic=EventTopic.RISK,
                    data={"action": "L1_WARNING", "drawdown": drawdown},
                    source="SENTINEL",
                    priority=Priority.HIGH,
                )
            )

        # L2: Halt — no new trades
        if drawdown >= SENTINEL.daily_equity_drawdown_l2 and not self._l2_triggered:
            self._l2_triggered = True
            logger.warning("SENTINEL L2 HALT: %.1f%% drawdown — trading halted", drawdown * 100)
            self._event_bus.publish_nowait(
                Event(
                    topic=EventTopic.RISK,
                    data={"action": "L2_HALT", "drawdown": drawdown},
                    source="SENTINEL",
                    priority=Priority.CRITICAL,
                )
            )

        # L3: Disconnect — emergency
        if drawdown >= SENTINEL.daily_equity_drawdown_l3 and not self._l3_triggered:
            self._l3_triggered = True
            logger.critical("SENTINEL L3 DISCONNECT: %.1f%% drawdown — EMERGENCY", drawdown * 100)
            self._event_bus.publish_nowait(
                Event(
                    topic=EventTopic.RISK,
                    data={"action": "L3_DISCONNECT", "drawdown": drawdown},
                    source="SENTINEL",
                    priority=Priority.CRITICAL,
                )
            )

    def register_position(self, position: Position) -> None:
        self._positions[position.position_id] = position
        self._trade_count_today += 1

    def close_position(self, position_id: str, exit_price: float) -> None:
        position = self._positions.pop(position_id, None)
        if position is None:
            return

        # FIXED: include lot_size_oz multiplier (1 lot = 100 oz for XAU/USD)
        lot_multiplier = SENTINEL.lot_size_oz
        if position.direction == "LONG":
            realized_pnl = (exit_price - position.entry_price) * position.size_lots * lot_multiplier
        else:
            realized_pnl = (position.entry_price - exit_price) * position.size_lots * lot_multiplier

        self._daily_pnl += realized_pnl

    def get_open_position_count(self) -> int:
        return len(self._positions)

    def get_open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_total_exposure_pct(self) -> float:
        return sum(position.size_pct for position in self._positions.values())

    def is_trading_allowed(self, direction: str = "LONG", strategy: str = "") -> bool:
        if self._l3_triggered:
            return False
        if self._l2_triggered:
            return False  # L2 = no new trades
        if self._clock.is_news_lockout():
            return False
        if self._clock.is_friday_lockout():
            return False
        if not self._clock.is_market_open():
            return False
        # v2: Check cascade, latency, correlation
        allowed, reason = self._v2.is_trade_allowed(direction, strategy)
        if not allowed:
            logger.warning("SENTINEL v2 blocked trade: %s", reason)
            return False
        return True

    def get_size_multiplier(self) -> float:
        """Position size multiplier based on risk level. L1 = 50% reduction."""
        if self._l1_triggered:
            return 0.5
        return 1.0

    def compute_position_size(self, base_size_pct: float) -> float:
        """v2: Compute regime & vol adjusted position size."""
        l1_mult = self.get_size_multiplier()
        v2_size = self._v2.compute_size(
            base_size_pct, self._regime, self._current_atr, self._avg_atr
        )
        return v2_size * l1_mult

    # ── v2 Integration Methods ───────────────────────────────────────────

    def record_latency(self, latency_ms: float, operation: str = "") -> None:
        self._v2.latency_monitor.record(latency_ms, operation)

    def report_module_failure(self, module_name: str, error_msg: str = "") -> bool:
        return self._v2.cascade_protection.report_failure(module_name, error_msg)

    def register_v2_position(self, position_id: str, direction: str, strategy: str = "") -> None:
        self._v2.correlation_guard.register_position(position_id, direction, strategy)

    def close_v2_position(self, position_id: str) -> None:
        self._v2.correlation_guard.remove_position(position_id)

    def set_regime(self, regime: str) -> None:
        self._regime = regime

    def set_atr(self, current_atr: float, avg_atr: float) -> None:
        self._current_atr = current_atr
        self._avg_atr = avg_atr

    @property
    def v2(self) -> SentinelV2:
        return self._v2

    def get_status(self) -> dict:
        drawdown = 0.0
        if self._session_peak_equity > 0:
            drawdown = (
                self._session_peak_equity - self._account_equity
            ) / self._session_peak_equity

        return {
            "l1_triggered": self._l1_triggered,
            "l2_triggered": self._l2_triggered,
            "l3_triggered": self._l3_triggered,
            "open_positions": self.get_open_position_count(),
            "total_exposure_pct": self.get_total_exposure_pct(),
            "daily_pnl": self._daily_pnl,
            "trade_count_today": self._trade_count_today,
            "account_equity": self._account_equity,
            "session_peak_equity": self._session_peak_equity,
            "trading_allowed": self.is_trading_allowed(),
            "current_drawdown_pct": drawdown,
            "size_multiplier": self.get_size_multiplier(),
        }

    async def _on_risk_event(self, event: Event) -> None:
        logger.debug("SENTINEL received RISK event: %s", event.data)
