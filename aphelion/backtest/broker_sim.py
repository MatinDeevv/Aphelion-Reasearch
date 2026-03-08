"""
APHELION Broker Simulator
Simulates a real broker with spread, slippage, commission, and SENTINEL validation.
Every order passes through TradeValidator before acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from aphelion.core.data_layer import Bar
from aphelion.backtest.order import (
    Order, OrderType, OrderSide, OrderStatus, Fill,
)
from aphelion.risk.sentinel.core import Position, SentinelCore
from aphelion.risk.sentinel.validator import TradeProposal, TradeValidator


@dataclass
class BrokerConfig:
    spread_pips: float = 0.30           # XAU/USD typical spread
    commission_per_lot: float = 7.0     # $ per lot (round trip / 2)
    slippage_pips: float = 0.10         # Average slippage on market orders
    slippage_std_pips: float = 0.05     # Std dev of slippage (random)
    max_slippage_pips: float = 0.50     # Maximum slippage cap
    pip_size: float = 0.01              # XAU pip = $0.01
    lot_size: float = 100.0             # 100 oz per standard lot
    min_lot: float = 0.01
    max_lot: float = 100.0
    gap_slippage_multiplier: float = 3.0  # Slippage multiplier on gap opens


class BrokerSimulator:
    """Simulates order execution with realistic market friction."""

    def __init__(
        self,
        config: BrokerConfig,
        validator: TradeValidator,
        sentinel_core: SentinelCore,
        rng: np.random.Generator,
    ):
        self._config = config
        self._validator = validator
        self._sentinel_core = sentinel_core
        self._rng = rng
        self._fills: list[Fill] = []
        self._rejected_orders: list[dict] = []
        self._bar_index: int = 0

    # ── Market Orders ────────────────────────────────────────────────────────

    def submit_market_order(
        self,
        order: Order,
        current_bar: Bar,
        account_equity: float,
    ) -> tuple[Order, Optional[Fill]]:
        """Submit a market order — validate through SENTINEL first."""
        if not self._validate_trade_proposal(order, current_bar, current_bar.close):
            order.status = OrderStatus.REJECTED
            return (order, None)

        # Compute fill price with spread + slippage
        cfg = self._config
        base_spread = cfg.spread_pips * cfg.pip_size
        raw_slippage = self._rng.normal(cfg.slippage_pips, cfg.slippage_std_pips)
        slippage = float(np.clip(raw_slippage, 0.0, cfg.max_slippage_pips))
        slippage_dollars = slippage * cfg.pip_size

        if order.side == OrderSide.BUY:
            fill_price = current_bar.close + base_spread + slippage_dollars
        else:
            fill_price = current_bar.close - base_spread - slippage_dollars

        commission = cfg.commission_per_lot * order.size_lots

        # Update order state
        order.filled_price = fill_price
        order.commission = commission
        order.slippage = slippage_dollars
        order.status = OrderStatus.FILLED
        order.filled_time = (
            current_bar.timestamp
            if isinstance(current_bar.timestamp, datetime)
            else datetime.now(timezone.utc)
        )

        # Create fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_price=fill_price,
            size_lots=order.size_lots,
            commission=commission,
            slippage_cost=slippage_dollars * order.size_lots * cfg.lot_size,
            fill_time=order.filled_time,
            bar_index=self._bar_index,
        )
        self._fills.append(fill)

        # Register position on sentinel_core
        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
        position = Position(
            position_id=order.order_id,
            symbol=order.symbol,
            direction=direction,
            entry_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            size_lots=order.size_lots,
            size_pct=order.size_pct,
            open_time=order.filled_time,
        )
        self._sentinel_core.register_position(position)

        return (order, fill)

    # ── Pending Orders ───────────────────────────────────────────────────────

    def check_pending_orders(
        self,
        pending_orders: list[Order],
        current_bar: Bar,
        account_equity: float,
    ) -> list[tuple[Order, Optional[Fill]]]:
        """Check pending LIMIT/STOP orders against current bar prices."""
        results: list[tuple[Order, Optional[Fill]]] = []

        for order in pending_orders:
            order.bars_alive += 1

            # Expire if too old
            if order.bars_alive > order.expiry_bars:
                order.status = OrderStatus.EXPIRED
                results.append((order, None))
                continue

            triggered = False
            fill_price = order.entry_price

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and current_bar.low <= order.entry_price:
                    triggered = True
                    fill_price = order.entry_price
                elif order.side == OrderSide.SELL and current_bar.high >= order.entry_price:
                    triggered = True
                    fill_price = order.entry_price

            elif order.order_type == OrderType.STOP:
                cfg = self._config
                raw_slippage = self._rng.normal(cfg.slippage_pips, cfg.slippage_std_pips)
                slippage = float(np.clip(raw_slippage, 0.0, cfg.max_slippage_pips))
                slippage_dollars = slippage * cfg.pip_size

                if order.side == OrderSide.BUY and current_bar.high >= order.entry_price:
                    triggered = True
                    fill_price = order.entry_price + slippage_dollars
                elif order.side == OrderSide.SELL and current_bar.low <= order.entry_price:
                    triggered = True
                    fill_price = order.entry_price - slippage_dollars

            if triggered:
                fill = self._submit_fill_at_price(
                    order, fill_price, current_bar, account_equity,
                )
                results.append((order, fill))
            else:
                results.append((order, None))

        return results

    def _submit_fill_at_price(
        self,
        order: Order,
        price: float,
        current_bar: Bar,
        account_equity: float,
    ) -> Optional[Fill]:
        """Fill an order at a specific price."""
        if not self._validate_trade_proposal(order, current_bar, price):
            order.status = OrderStatus.REJECTED
            return None

        cfg = self._config
        commission = cfg.commission_per_lot * order.size_lots

        order.filled_price = price
        order.commission = commission
        order.slippage = abs(price - order.entry_price)
        order.status = OrderStatus.FILLED
        order.filled_time = (
            current_bar.timestamp
            if isinstance(current_bar.timestamp, datetime)
            else datetime.now(timezone.utc)
        )

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_price=price,
            size_lots=order.size_lots,
            commission=commission,
            slippage_cost=order.slippage * order.size_lots * cfg.lot_size,
            fill_time=order.filled_time,
            bar_index=self._bar_index,
        )
        self._fills.append(fill)

        # Register position on sentinel_core
        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
        position = Position(
            position_id=order.order_id,
            symbol=order.symbol,
            direction=direction,
            entry_price=price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            size_lots=order.size_lots,
            size_pct=order.size_pct,
            open_time=order.filled_time,
        )
        self._sentinel_core.register_position(position)

        return fill

    # ── SL/TP Checking ───────────────────────────────────────────────────────

    def _validate_trade_proposal(
        self, order: Order, current_bar: Bar, entry_price: float,
    ) -> bool:
        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
        proposal = TradeProposal(
            symbol=order.symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            size_pct=order.size_pct,
            proposed_by=order.proposed_by,
        )
        result = self._validator.validate(proposal)
        if result.approved:
            return True

        self._rejected_orders.append({
            "order_id": order.order_id,
            "reason": " | ".join(result.rejections),
            "time": current_bar.timestamp.isoformat()
            if isinstance(current_bar.timestamp, datetime)
            else str(current_bar.timestamp),
        })
        return False

    def check_sl_tp(
        self,
        open_positions: list[Position],
        current_bar: Bar,
    ) -> list[tuple[str, float, str]]:
        """Check if any position's SL or TP was hit on this bar."""
        exits: list[tuple[str, float, str]] = []
        cfg = self._config

        for pos in open_positions:
            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None

            if pos.direction == "LONG":
                # Gap check — open already past SL
                if current_bar.open < pos.stop_loss:
                    exit_price = current_bar.open
                    exit_reason = "SL_HIT"
                elif current_bar.low <= pos.stop_loss:
                    # SL hit within bar — apply slippage
                    raw_slip = self._rng.exponential(cfg.slippage_pips) * cfg.gap_slippage_multiplier
                    sl_slip = min(raw_slip, cfg.max_slippage_pips)
                    exit_price = pos.stop_loss - sl_slip * cfg.pip_size
                    exit_reason = "SL_HIT"
                elif current_bar.high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "TP_HIT"

            else:  # SHORT
                # Gap check — open already past SL
                if current_bar.open > pos.stop_loss:
                    exit_price = current_bar.open
                    exit_reason = "SL_HIT"
                elif current_bar.high >= pos.stop_loss:
                    raw_slip = self._rng.exponential(cfg.slippage_pips) * cfg.gap_slippage_multiplier
                    sl_slip = min(raw_slip, cfg.max_slippage_pips)
                    exit_price = pos.stop_loss + sl_slip * cfg.pip_size
                    exit_reason = "SL_HIT"
                elif current_bar.low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "TP_HIT"

            if exit_price is not None and exit_reason is not None:
                exits.append((pos.position_id, exit_price, exit_reason))

        return exits

    def set_bar_index(self, index: int) -> None:
        self._bar_index = index

    # ── Stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        total_comm = sum(f.commission for f in self._fills)
        total_slip = sum(f.slippage_cost for f in self._fills)
        n = len(self._fills)
        avg_slip = (
            sum(f.slippage_cost for f in self._fills) / n if n > 0 else 0.0
        )
        return {
            "fill_count": n,
            "rejection_count": len(self._rejected_orders),
            "total_commission": total_comm,
            "total_slippage_cost": total_slip,
            "avg_slippage_pips": avg_slip / (self._config.pip_size * self._config.lot_size)
            if n > 0
            else 0.0,
        }
