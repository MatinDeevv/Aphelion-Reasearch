"""
APHELION Backtest Order Types
Enums and dataclasses for order management, fills, and completed trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    size_lots: float
    entry_price: float              # Requested price (0.0 for MARKET)
    stop_loss: float
    take_profit: float
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    proposed_by: str = "SYSTEM"
    created_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_pct: float = 0.0           # Fraction of account when order was placed
    expiry_bars: int = 10           # Cancel after N bars if not filled (for LIMIT/STOP)
    bars_alive: int = 0             # Incremented each bar while pending


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    filled_price: float
    size_lots: float
    commission: float
    slippage_cost: float
    fill_time: datetime
    bar_index: int


@dataclass
class BacktestTrade:
    trade_id: str
    symbol: str
    direction: str              # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    size_lots: float
    size_pct: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    commission: float
    net_pnl: float
    exit_reason: str            # "SL_HIT", "TP_HIT", "MANUAL", "FRIDAY_CLOSE", "L3_HALT"
    bars_held: int
    proposed_by: str
    entry_bar_index: int
    exit_bar_index: int

    _LOT_SIZE_OZ: float = 100.0  # oz per standard lot (XAU/USD)

    @property
    def pnl_pct(self) -> float:
        """P&L as percentage of initial risk (dollar-risk denominator)."""
        risk_dollars = (
            abs(self.entry_price - self.stop_loss)
            * self.size_lots
            * self._LOT_SIZE_OZ
        )
        if risk_dollars == 0:
            return 0.0
        return (self.net_pnl / risk_dollars) * 100.0

    @property
    def r_multiple(self) -> float:
        """Net P&L expressed as a multiple of initial dollar risk."""
        risk = (
            abs(self.entry_price - self.stop_loss)
            * self.size_lots
            * self._LOT_SIZE_OZ  # Must match portfolio PnL calc
        )
        if risk == 0:
            return 0.0
        return self.net_pnl / risk
