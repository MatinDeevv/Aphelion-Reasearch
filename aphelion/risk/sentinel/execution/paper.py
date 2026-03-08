"""
APHELION Paper Trading Executor
Simulates order fills against live market prices without touching a real broker.
Tracks a virtual account with full SENTINEL enforcement.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from aphelion.backtest.order import (
    Fill, Order, OrderSide, OrderStatus, OrderType,
)
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import SENTINEL
from aphelion.core.event_bus import Event, EventBus, EventTopic, Priority
from aphelion.risk.sentinel.core import Position, SentinelCore
from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer
from aphelion.risk.sentinel.validator import TradeProposal

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PaperConfig:
    """Paper trading executor configuration."""
    initial_capital: float = 10_000.0
    slippage_points: float = 0.10          # Simulated slippage (XAU points)
    commission_per_lot: float = 7.0        # Round-trip commission per standard lot
    max_pending_ttl_seconds: float = 300   # Cancel pending orders after 5 min
    fill_latency_ms: float = 50.0          # Simulated fill delay (logging only)
    symbol: str = "XAUUSD"
    lot_size_oz: float = 100.0             # 1 standard lot = 100 troy oz


# ── Paper Fill ───────────────────────────────────────────────────────────────

@dataclass
class PaperFill:
    """Record of a simulated fill for audit purposes."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    filled_price: float
    requested_price: float
    slippage: float
    size_lots: float
    commission: float
    fill_time: datetime


# ── Executor ─────────────────────────────────────────────────────────────────

class PaperExecutor:
    """
    Simulates live order execution against current market prices.
    All orders pass through ExecutionEnforcer before virtual fill.
    Portfolio and SentinelCore are updated in lockstep.
    """

    def __init__(
        self,
        config: PaperConfig,
        enforcer: ExecutionEnforcer,
        sentinel_core: SentinelCore,
        portfolio: Portfolio,
        event_bus: EventBus,
    ):
        self._config = config
        self._enforcer = enforcer
        self._sentinel = sentinel_core
        self._portfolio = portfolio
        self._event_bus = event_bus

        self._pending_orders: list[Order] = []
        self._fill_history: list[PaperFill] = []
        self._rejection_count: int = 0
        self._fill_count: int = 0
        self._last_price: float = 0.0
        self._bar_index: int = 0

    # ── Price feed ────────────────────────────────────────────────────────

    def update_price(self, price: float, bar_index: int = 0) -> None:
        """Update current market price and bar index."""
        self._last_price = price
        self._bar_index = bar_index

    # ── Submit order ──────────────────────────────────────────────────────

    def submit_order(self, order: Order, current_price: float) -> Optional[PaperFill]:
        """
        Route an order through SENTINEL enforcement and simulate a fill.

        Returns PaperFill on success, None on rejection.
        """
        self.update_price(current_price)

        # Build TradeProposal for enforcer
        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
        entry_price = current_price if order.order_type == OrderType.MARKET else order.entry_price

        proposal = TradeProposal(
            symbol=order.symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            size_pct=order.size_pct,
            proposed_by=order.proposed_by,
        )

        # Enforce through SENTINEL pipeline
        approved, reason, final_size_pct = self._enforcer.approve_order(proposal)

        if not approved or final_size_pct <= 0:
            order.status = OrderStatus.REJECTED
            self._rejection_count += 1
            logger.info(
                "PAPER REJECTED order %s: %s",
                order.order_id, reason,
            )
            self._publish_event("ORDER_REJECTED", {
                "order_id": order.order_id,
                "reason": reason,
                "proposed_by": order.proposed_by,
            })
            return None

        # Adjust size if enforcer modified it
        if final_size_pct != order.size_pct and order.size_pct > 0:
            scale = final_size_pct / order.size_pct
            order.size_lots = max(0.01, round(order.size_lots * scale, 2))
        order.size_pct = final_size_pct

        # Handle order type
        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, current_price)
        else:
            order.created_time = datetime.now(timezone.utc)
            self._pending_orders.append(order)
            logger.info(
                "PAPER PENDING %s order %s @ %.2f",
                order.order_type.value, order.order_id, order.entry_price,
            )
            return None

    # ── Market fill ───────────────────────────────────────────────────────

    def _fill_market(self, order: Order, current_price: float) -> PaperFill:
        """Simulate a market order fill with slippage."""
        slippage = self._config.slippage_points
        if order.side == OrderSide.BUY:
            fill_price = current_price + slippage
        else:
            fill_price = current_price - slippage

        fill_price = round(fill_price, 2)
        commission = self._config.commission_per_lot * order.size_lots
        now = datetime.now(timezone.utc)

        # Update order state
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_time = now
        order.commission = commission
        order.slippage = slippage

        # Create fill record
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_price=fill_price,
            size_lots=order.size_lots,
            commission=commission,
            slippage_cost=slippage * order.size_lots * self._config.lot_size_oz,
            fill_time=now,
            bar_index=self._bar_index,
        )

        # Register with Portfolio and SentinelCore
        position = self._portfolio.open_position(fill, order)
        self._sentinel.register_position(position)

        # Track fill
        paper_fill = PaperFill(
            fill_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_price=fill_price,
            requested_price=current_price,
            slippage=slippage,
            size_lots=order.size_lots,
            commission=commission,
            fill_time=now,
        )
        self._fill_history.append(paper_fill)
        self._fill_count += 1

        logger.info(
            "PAPER FILL %s %s %.2f lots @ %.2f (slip=%.2f, comm=%.2f)",
            order.side.value, order.symbol,
            order.size_lots, fill_price, slippage, commission,
        )

        self._publish_event("ORDER_FILLED", {
            "order_id": order.order_id,
            "fill_price": fill_price,
            "size_lots": order.size_lots,
            "side": order.side.value,
            "commission": commission,
        })

        return paper_fill

    # ── Pending order management ──────────────────────────────────────────

    def check_pending_orders(self, current_price: float) -> list[PaperFill]:
        """Check and fill any triggered pending orders. Returns list of fills."""
        self.update_price(current_price)
        fills: list[PaperFill] = []
        still_pending: list[Order] = []
        now = datetime.now(timezone.utc)

        for order in self._pending_orders:
            # Check TTL expiry
            age = (now - order.created_time).total_seconds()
            if age > self._config.max_pending_ttl_seconds:
                order.status = OrderStatus.EXPIRED
                logger.info("PAPER EXPIRED order %s after %.0fs", order.order_id, age)
                continue

            triggered = self._is_pending_triggered(order, current_price)
            if triggered:
                fill = self._fill_market(order, order.entry_price)
                if fill:
                    fills.append(fill)
            else:
                still_pending.append(order)

        self._pending_orders = still_pending
        return fills

    def _is_pending_triggered(self, order: Order, price: float) -> bool:
        """Check if a pending order's trigger condition is met."""
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and price <= order.entry_price:
                return True
            if order.side == OrderSide.SELL and price >= order.entry_price:
                return True
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY and price >= order.entry_price:
                return True
            if order.side == OrderSide.SELL and price <= order.entry_price:
                return True
        return False

    # ── Position exit ─────────────────────────────────────────────────────

    def check_sl_tp(self, current_price: float) -> list[tuple[str, float, str]]:
        """
        Check all open positions for SL/TP hits.
        Returns list of (position_id, exit_price, reason).
        """
        self.update_price(current_price)
        exits: list[tuple[str, float, str]] = []

        for pos in self._sentinel.get_open_positions():
            if pos.direction == "LONG":
                if current_price <= pos.stop_loss:
                    exits.append((pos.position_id, pos.stop_loss, "SL_HIT"))
                elif current_price >= pos.take_profit:
                    exits.append((pos.position_id, pos.take_profit, "TP_HIT"))
            else:  # SHORT
                if current_price >= pos.stop_loss:
                    exits.append((pos.position_id, pos.stop_loss, "SL_HIT"))
                elif current_price <= pos.take_profit:
                    exits.append((pos.position_id, pos.take_profit, "TP_HIT"))

        return exits

    def close_position(
        self, position_id: str, exit_price: float, reason: str,
    ) -> Optional[float]:
        """
        Close a position at the given price. Returns net P&L or None.
        """
        now = datetime.now(timezone.utc)
        commission = 0.0
        pos = next(
            (p for p in self._sentinel.get_open_positions() if p.position_id == position_id),
            None,
        )
        if pos:
            commission = self._config.commission_per_lot * pos.size_lots

        trade = self._portfolio.close_position(
            position_id, exit_price, now, reason, self._bar_index, commission,
        )
        self._sentinel.close_position(position_id, exit_price)

        if trade:
            logger.info(
                "PAPER CLOSE %s P&L=%.2f reason=%s",
                position_id, trade.net_pnl, reason,
            )
            self._publish_event("POSITION_CLOSED", {
                "position_id": position_id,
                "exit_price": exit_price,
                "reason": reason,
                "net_pnl": trade.net_pnl,
            })
            return trade.net_pnl
        return None

    def force_close_all(self, current_price: float, reason: str) -> int:
        """Force close all open positions. Returns count closed."""
        positions = list(self._sentinel.get_open_positions())
        count = 0
        for pos in positions:
            self.close_position(pos.position_id, current_price, reason)
            count += 1
        if count:
            logger.warning("PAPER force-closed %d positions: %s", count, reason)
        return count

    # ── Event publishing ──────────────────────────────────────────────────

    def _publish_event(self, action: str, data: dict) -> None:
        """Publish a paper trading event to the event bus."""
        self._event_bus.publish_nowait(Event(
            topic=EventTopic.RISK,
            data={"action": action, **data},
            source="PAPER_EXECUTOR",
            priority=Priority.NORMAL,
        ))

    # ── Stats ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "fill_count": self._fill_count,
            "rejection_count": self._rejection_count,
            "pending_count": len(self._pending_orders),
            "fill_history_len": len(self._fill_history),
            "last_price": self._last_price,
        }

    @property
    def fill_history(self) -> list[PaperFill]:
        return list(self._fill_history)
