"""
APHELION Backtest Engine
Event-driven bar-by-bar simulation engine. Iterates historical data,
generates strategy signals, runs SENTINEL validation, simulates order
execution, and records full results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

import numpy as np

from aphelion.backtest.broker_sim import BrokerConfig, BrokerSimulator
from aphelion.backtest.order import (
    BacktestTrade, Order, OrderType, OrderSide, OrderStatus,
)
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar, DataLayer
from aphelion.core.clock import MarketClock
from aphelion.features.engine import FeatureEngine
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
from aphelion.risk.sentinel.core import SentinelCore
from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer
from aphelion.risk.sentinel.position_sizer import PositionSizer
from aphelion.risk.sentinel.validator import TradeProposal, TradeValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "XAUUSD"
    timeframe: Timeframe = Timeframe.M1
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.02
    broker_config: BrokerConfig = field(default_factory=BrokerConfig)
    max_bars: Optional[int] = None
    warmup_bars: int = 50
    random_seed: int = 42
    enable_feature_engine: bool = True


@dataclass
class BacktestResults:
    config: BacktestConfig
    trades: list[BacktestTrade]
    equity_curve: tuple[list[datetime], list[float]]
    drawdown_curve: tuple[list[datetime], list[float]]
    total_bars: int
    warmup_bars: int
    broker_stats: dict
    sentinel_rejections: int
    final_equity: float
    initial_capital: float
    daily_returns: list[float]

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100


class BacktestEngine:
    """
    Main event-driven backtest engine. Iterates bar-by-bar through historical
    data, calling strategy signals, running SENTINEL validation, simulating
    order execution, and recording everything.
    """

    def __init__(
        self,
        config: BacktestConfig,
        sentinel_stack: dict,
        data_layer: DataLayer,
    ):
        """
        Args:
            config: Backtest configuration.
            sentinel_stack: Dict with keys: core, validator, cb, enforcer, sizer.
            data_layer: DataLayer instance for feature computation.
        """
        self._config = config

        # Unpack sentinel stack
        self._sentinel_core: SentinelCore = sentinel_stack["core"]
        self._validator: TradeValidator = sentinel_stack["validator"]
        self._cb: CircuitBreaker = sentinel_stack["cb"]
        self._enforcer: ExecutionEnforcer = sentinel_stack["enforcer"]
        self._sizer: PositionSizer = sentinel_stack["sizer"]
        self._clock: MarketClock = sentinel_stack["clock"]  # FIXED: store clock for simulated time
        self._data_layer = data_layer

        self._strategy_callback: Optional[Callable] = None
        self._pending_orders: list[Order] = []
        self._bar_index: int = 0
        self._results: Optional[BacktestResults] = None
        self._feature_engine: Optional[FeatureEngine] = None
        self._reset_runtime_state()

    # ── Strategy ─────────────────────────────────────────────────────────────

    def set_strategy(self, callback: Callable) -> None:
        """Set strategy callback: (bar, features, portfolio) -> list[Order]."""
        self._strategy_callback = callback

    # ── Main Loop ────────────────────────────────────────────────────────────

    def run(self, bars: list[Bar]) -> BacktestResults:
        """Run the backtest over a list of bars."""
        self._reset_runtime_state()

        if self._config.max_bars is not None:
            bars = bars[: self._config.max_bars]

        total_bars = len(bars)

        for i, bar in enumerate(bars):
            self._bar_index = i
            self._broker.set_bar_index(i)

            # Bar integrity check — skip corrupted bars
            if (
                bar.close <= 0
                or bar.high < bar.low
                or any(
                    v != v  # NaN check (x != x is True for NaN)
                    for v in (bar.open, bar.high, bar.low, bar.close)
                )
            ):
                self._portfolio.update_bar(bar, i)
                continue

            # FIXED: Set simulated time so MarketClock uses bar timestamp
            bar_ts = bar.timestamp if isinstance(bar.timestamp, datetime) else None
            if bar_ts is not None:
                self._clock.set_simulated_time(bar_ts)

            # 1. Update sentinel equity
            self._sentinel_core.update_equity(self._portfolio.equity)

            # 2. Update circuit breaker
            self._cb.update(self._portfolio.equity)

            # 3. Handle L3 halt — force close everything
            if self._sentinel_core.l3_triggered:
                self._handle_l3_halt(bar)

            # 4. Handle Friday close
            self._handle_friday_close(bar)

            # 5. Check SL/TP on open positions
            open_positions = self._sentinel_core.get_open_positions()
            sl_tp_exits = self._broker.check_sl_tp(open_positions, bar)

            for position_id, exit_price, reason in sl_tp_exits:
                exit_time = (
                    bar.timestamp
                    if isinstance(bar.timestamp, datetime)
                    else datetime.now(timezone.utc)
                )
                commission = self._config.broker_config.commission_per_lot
                # Get the position to find size_lots for commission
                pos = next(
                    (p for p in open_positions if p.position_id == position_id),
                    None,
                )
                exit_comm = (
                    self._config.broker_config.commission_per_lot * pos.size_lots
                    if pos
                    else 0.0
                )
                self._portfolio.close_position(
                    position_id, exit_price, exit_time, reason, i, exit_comm,
                )
                self._sentinel_core.close_position(position_id, exit_price)

            # 6. Check and process pending limit/stop orders
            if self._pending_orders:
                pending_results = self._broker.check_pending_orders(
                    self._pending_orders, bar, self._portfolio.equity,
                )
                still_pending: list[Order] = []
                for order, fill in pending_results:
                    if fill is not None:
                        self._portfolio.open_position(fill, order)
                    elif order.status == OrderStatus.PENDING:
                        still_pending.append(order)
                    # EXPIRED / CANCELLED orders are dropped
                self._pending_orders = still_pending

            # 7. Compute features
            features: dict = {}
            if self._feature_engine is not None:
                try:
                    features = self._feature_engine.on_bar(bar)
                except Exception:
                    logger.debug("Feature engine error on bar %d", i, exc_info=True)

            # 8. Strategy callback (only after warmup)
            if i >= self._config.warmup_bars and self._strategy_callback is not None:
                try:
                    new_orders = self._strategy_callback(bar, features, self._portfolio)
                    if new_orders:
                        for order in new_orders:
                            order = self._apply_execution_enforcement(order, bar)
                            if order is None:
                                self._enforcer_rejections += 1  # FIXED: track enforcer rejections
                                continue

                            if order.order_type == OrderType.MARKET:
                                filled_order, fill = self._broker.submit_market_order(
                                    order, bar, self._portfolio.equity,
                                )
                                if fill:
                                    self._portfolio.open_position(fill, filled_order)
                            else:
                                self._pending_orders.append(order)
                except Exception:
                    logger.debug("Strategy error on bar %d", i, exc_info=True)

            # 9. Update portfolio with this bar
            self._portfolio.update_bar(bar, i)

        # Build results
        # FIXED: Reset simulated time after run completes
        self._clock.set_simulated_time(None)

        eq_ts, eq_vals = self._portfolio.get_equity_series()
        dd_ts, dd_vals = self._portfolio.get_drawdown_series()

        # FIXED: Count both enforcer-level and broker-level rejections
        total_rejections = self._broker.stats["rejection_count"] + self._enforcer_rejections

        self._results = BacktestResults(
            config=self._config,
            trades=self._portfolio.get_closed_trades(),
            equity_curve=(eq_ts, eq_vals),
            drawdown_curve=(dd_ts, dd_vals),
            total_bars=total_bars,
            warmup_bars=self._config.warmup_bars,
            broker_stats=self._broker.stats,
            sentinel_rejections=total_rejections,
            final_equity=self._portfolio.equity,
            initial_capital=self._config.initial_capital,
            daily_returns=self._portfolio.get_daily_returns(),
        )
        return self._results

    def _reset_runtime_state(self) -> None:
        """
        Reset per-run engine state so repeated runs are deterministic and isolated.
        """
        self._reset_sentinel_runtime()
        self._rng = np.random.default_rng(self._config.random_seed)
        self._broker = BrokerSimulator(
            config=self._config.broker_config,
            validator=self._validator,
            sentinel_core=self._sentinel_core,
            rng=self._rng,
        )
        self._portfolio = Portfolio(self._config.initial_capital)
        self._pending_orders = []
        self._bar_index = 0
        self._enforcer_rejections = 0  # FIXED: track enforcer-level rejections
        self._results = None
        if self._config.enable_feature_engine:
            self._feature_engine = FeatureEngine(self._data_layer)
        else:
            self._feature_engine = None

    def _reset_sentinel_runtime(self) -> None:
        """
        Best-effort reset of mutable SENTINEL runtime state for backtest repeatability.
        """
        try:
            if hasattr(self._sentinel_core, "_positions"):
                self._sentinel_core._positions.clear()  # noqa: SLF001
            if hasattr(self._sentinel_core, "_daily_pnl"):
                self._sentinel_core._daily_pnl = 0.0  # noqa: SLF001
            if hasattr(self._sentinel_core, "_trade_count_today"):
                self._sentinel_core._trade_count_today = 0  # noqa: SLF001
            if hasattr(self._sentinel_core, "_l1_triggered"):
                self._sentinel_core._l1_triggered = False  # noqa: SLF001
            if hasattr(self._sentinel_core, "_l2_triggered"):
                self._sentinel_core._l2_triggered = False  # noqa: SLF001
            if hasattr(self._sentinel_core, "_l3_triggered"):
                self._sentinel_core._l3_triggered = False  # noqa: SLF001
            if hasattr(self._sentinel_core, "_account_equity"):
                self._sentinel_core._account_equity = self._config.initial_capital  # noqa: SLF001
            if hasattr(self._sentinel_core, "_session_peak_equity"):
                self._sentinel_core._session_peak_equity = self._config.initial_capital  # noqa: SLF001
        except Exception:
            logger.debug("Unable to fully reset SentinelCore state", exc_info=True)

        try:
            if hasattr(self._cb, "_state"):
                self._cb._state = "NORMAL"  # noqa: SLF001
            if hasattr(self._cb, "_size_multiplier"):
                self._cb._size_multiplier = 1.0  # noqa: SLF001
            if hasattr(self._cb, "_peak_equity"):
                self._cb._peak_equity = 0.0  # noqa: SLF001
            if hasattr(self._cb, "_current_equity"):
                self._cb._current_equity = 0.0  # noqa: SLF001
            if hasattr(self._cb, "_triggers"):
                self._cb._triggers.clear()  # noqa: SLF001
        except Exception:
            logger.debug("Unable to fully reset CircuitBreaker state", exc_info=True)

        try:
            if hasattr(self._enforcer, "_approved_count"):
                self._enforcer._approved_count = 0  # noqa: SLF001
            if hasattr(self._enforcer, "_rejected_count"):
                self._enforcer._rejected_count = 0  # noqa: SLF001
            if hasattr(self._enforcer, "_rejection_log"):
                self._enforcer._rejection_log.clear()  # noqa: SLF001
        except Exception:
            logger.debug("Unable to fully reset ExecutionEnforcer state", exc_info=True)

    # ── Friday Close ─────────────────────────────────────────────────────────

    def _handle_friday_close(self, bar: Bar) -> None:
        """Force-close all positions 30 min before Friday market close."""
        ts = bar.timestamp if isinstance(bar.timestamp, datetime) else None
        if ts is None:
            return

        if ts.weekday() != 4:  # Not Friday
            return

        close_time = ts.replace(hour=21, minute=0, second=0, microsecond=0)
        lockout_start = close_time - timedelta(minutes=30)

        if ts >= lockout_start:
            self._force_close_all(bar, "FRIDAY_CLOSE")

    # ── L3 Halt ──────────────────────────────────────────────────────────────

    def _handle_l3_halt(self, bar: Bar) -> None:
        """Force close all positions on L3 drawdown disconnect."""
        self._force_close_all(bar, "L3_HALT")
        self._pending_orders.clear()

    # ── Force Close ──────────────────────────────────────────────────────────

    def _force_close_all(self, bar: Bar, reason: str) -> None:
        """Close all open positions at bar close."""
        open_positions = list(self._sentinel_core.get_open_positions())
        exit_time = (
            bar.timestamp
            if isinstance(bar.timestamp, datetime)
            else datetime.now(timezone.utc)
        )
        for pos in open_positions:
            exit_comm = self._config.broker_config.commission_per_lot * pos.size_lots
            self._portfolio.close_position(
                pos.position_id, bar.close, exit_time, reason,
                self._bar_index, exit_comm,
            )
            self._sentinel_core.close_position(pos.position_id, bar.close)

    # ── Accessors ────────────────────────────────────────────────────────────

    def _apply_execution_enforcement(self, order: Order, bar: Bar) -> Optional[Order]:
        """
        Apply the Phase 2 enforcement pipeline before broker execution.
        This keeps backtest behavior aligned with live SENTINEL rules.
        """
        if order.order_type == OrderType.MARKET:
            proposed_entry = bar.close
        else:
            proposed_entry = order.entry_price if order.entry_price > 0 else bar.close

        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
        proposal = TradeProposal(
            symbol=order.symbol,
            direction=direction,
            entry_price=proposed_entry,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            size_pct=order.size_pct,
            proposed_by=order.proposed_by,
        )
        approved, _, final_size_pct = self._enforcer.approve_order(proposal)
        if not approved or final_size_pct <= 0:
            order.status = OrderStatus.REJECTED
            return None

        if order.size_pct > 0 and final_size_pct != order.size_pct:
            scale = final_size_pct / order.size_pct
            order.size_lots = max(0.01, round(order.size_lots * scale, 2))

        order.size_pct = final_size_pct
        return order

    def get_results(self) -> Optional[BacktestResults]:
        return self._results
