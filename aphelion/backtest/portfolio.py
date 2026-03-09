"""
APHELION Portfolio Tracker
Tracks account equity, positions, running P&L, equity curve, and drawdown
during backtests. Source of truth for all financial state in simulation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from aphelion.backtest.order import BacktestTrade, Fill, Order, OrderSide
from aphelion.core.data_layer import Bar
from aphelion.risk.sentinel.core import Position


# XAU/USD: 1 standard lot = 100 oz
_LOT_SIZE_OZ: float = 100.0


class Portfolio:
    """Tracks complete financial state during a backtest run."""

    def __init__(self, initial_capital: float):
        self._initial_capital = initial_capital
        self._cash: float = initial_capital
        self._equity: float = initial_capital
        self._peak_equity: float = initial_capital
        self._equity_curve: list[float] = [initial_capital]
        self._drawdown_curve: list[float] = [0.0]
        self._daily_equity: dict[str, float] = {}
        self._trades: list[BacktestTrade] = []
        self._open_positions: dict[str, Position] = {}
        self._bar_timestamps: list[datetime] = []
        self._trade_counter: int = 0
        # Track entry info not stored on Position
        self._position_meta: dict[str, dict] = {}  # id → {bar_index, order}

    # ── Trade IDs ────────────────────────────────────────────────────────────

    def generate_trade_id(self) -> str:
        tid = f"BT-{self._trade_counter:06d}"
        self._trade_counter += 1
        return tid

    # ── Position Management ──────────────────────────────────────────────────

    def open_position(self, fill: Fill, order: Order) -> Position:
        """Create and store a position from a filled order."""
        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
        position = Position(
            position_id=order.order_id,
            symbol=order.symbol,
            direction=direction,
            entry_price=fill.filled_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            size_lots=fill.size_lots,
            size_pct=order.size_pct,
            open_time=fill.fill_time,
        )
        self._open_positions[position.position_id] = position
        self._position_meta[position.position_id] = {
            "bar_index": fill.bar_index,
            "proposed_by": order.proposed_by,
            "commission_entry": fill.commission,
        }
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        bar_index: int,
        commission: float = 0.0,
    ) -> Optional[BacktestTrade]:
        """Close a position and record the completed trade."""
        position = self._open_positions.pop(position_id, None)
        if position is None:
            return None

        meta = self._position_meta.pop(position_id, {})
        entry_bar = meta.get("bar_index", 0)
        proposed_by = meta.get("proposed_by", "SYSTEM")
        entry_commission = meta.get("commission_entry", 0.0)

        # Gross P&L: (exit - entry) * lots * 100 for LONG, (entry - exit) for SHORT
        if position.direction == "LONG":
            gross_pnl = (exit_price - position.entry_price) * position.size_lots * _LOT_SIZE_OZ
        else:
            gross_pnl = (position.entry_price - exit_price) * position.size_lots * _LOT_SIZE_OZ

        total_commission = entry_commission + commission
        net_pnl = gross_pnl - total_commission

        self._cash += net_pnl

        trade = BacktestTrade(
            trade_id=self.generate_trade_id(),
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size_lots=position.size_lots,
            size_pct=position.size_pct,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            entry_time=position.open_time,
            exit_time=exit_time,
            gross_pnl=gross_pnl,
            commission=total_commission,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            bars_held=bar_index - entry_bar,
            proposed_by=proposed_by,
            entry_bar_index=entry_bar,
            exit_bar_index=bar_index,
        )
        self._trades.append(trade)
        return trade

    # ── Bar Update ───────────────────────────────────────────────────────────

    def update_bar(self, bar: Bar, bar_index: int) -> None:
        """Update unrealized P&L and equity for all open positions."""
        unrealized = 0.0
        for pos in self._open_positions.values():
            if pos.direction == "LONG":
                unrealized += (bar.close - pos.entry_price) * pos.size_lots * _LOT_SIZE_OZ
            else:
                unrealized += (pos.entry_price - bar.close) * pos.size_lots * _LOT_SIZE_OZ

        self._equity = self._cash + unrealized

        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        self._equity_curve.append(self._equity)

        dd = 0.0
        if self._peak_equity > 0:
            dd = (self._peak_equity - self._equity) / self._peak_equity
        self._drawdown_curve.append(dd)

        ts = bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now(tz=timezone.utc)
        self._bar_timestamps.append(ts)

        date_str = ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts)
        self._daily_equity[date_str] = self._equity

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self._equity) / self._peak_equity

    @property
    def total_return(self) -> float:
        return (self._equity - self._initial_capital) / self._initial_capital

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_closed_trades(self) -> list[BacktestTrade]:
        return list(self._trades)

    def get_equity_series(self) -> tuple[list[datetime], list[float]]:
        if len(self._equity_curve) == len(self._bar_timestamps) + 1:
            if self._bar_timestamps:
                return [self._bar_timestamps[0], *self._bar_timestamps], self._equity_curve
            return [], self._equity_curve
        return self._bar_timestamps, self._equity_curve

    def get_drawdown_series(self) -> tuple[list[datetime], list[float]]:
        if len(self._drawdown_curve) == len(self._bar_timestamps) + 1:
            if self._bar_timestamps:
                return [self._bar_timestamps[0], *self._bar_timestamps], self._drawdown_curve
            return [], self._drawdown_curve
        return self._bar_timestamps, self._drawdown_curve

    def get_daily_returns(self) -> list[float]:
        dates = sorted(self._daily_equity.keys())
        if len(dates) < 2:
            return []
        daily_equities = [self._daily_equity[d] for d in dates]
        returns: list[float] = []
        for i in range(1, len(daily_equities)):
            prev = daily_equities[i - 1]
            curr = daily_equities[i]
            if prev > 0:
                returns.append((curr - prev) / prev)
        return returns

    def get_exposure_pct(self) -> float:
        return sum(p.size_pct for p in self._open_positions.values())
