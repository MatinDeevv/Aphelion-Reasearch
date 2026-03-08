"""SENTINEL core risk authority and hard-limit state tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from aphelion.core.clock import MarketClock
from aphelion.core.config import EventTopic, SENTINEL
from aphelion.core.event_bus import Event, EventBus, Priority

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
    """Supreme, immutable risk authority for live trading controls."""

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
        self._event_bus.subscribe(EventTopic.RISK, self._on_risk_event)

    @property
    def l3_triggered(self) -> bool:
        return self._l3_triggered

    def update_equity(self, equity: float) -> None:
        self._account_equity = equity

        if equity > self._session_peak_equity:
            self._session_peak_equity = equity

        drawdown = 0.0
        if self._session_peak_equity > 0:
            drawdown = (self._session_peak_equity - equity) / self._session_peak_equity

        if drawdown >= SENTINEL.daily_equity_drawdown_l3 and not self._l3_triggered:
            self._l3_triggered = True
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

    def close_position(self, position_id: str, exit_price: float) -> None:
        position = self._positions.pop(position_id, None)
        if position is None:
            return

        if position.direction == "LONG":
            realized_pnl = (exit_price - position.entry_price) * position.size_lots
        else:
            realized_pnl = (position.entry_price - exit_price) * position.size_lots

        self._daily_pnl += realized_pnl

    def get_open_position_count(self) -> int:
        return len(self._positions)

    def get_open_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_total_exposure_pct(self) -> float:
        return sum(position.size_pct for position in self._positions.values())

    def is_trading_allowed(self) -> bool:
        if self._l3_triggered:
            return False
        if self._clock.is_news_lockout():
            return False
        if self._clock.is_friday_lockout():
            return False
        if not self._clock.is_market_open():
            return False
        return True

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
            "account_equity": self._account_equity,
            "session_peak_equity": self._session_peak_equity,
            "trading_allowed": self.is_trading_allowed(),
            "current_drawdown_pct": drawdown,
        }

    async def _on_risk_event(self, event: Event) -> None:
        logger.debug("SENTINEL received RISK event: %s", event.data)
