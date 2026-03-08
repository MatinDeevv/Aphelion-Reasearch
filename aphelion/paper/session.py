"""
APHELION Paper Trading Session Orchestrator
Async main loop that wires:
  DataFeed → FeatureEngine → HYDRA → Strategy → SENTINEL → PaperExecutor → Ledger

Supports three data-feed modes: LIVE (MT5), REPLAY, and SIMULATED.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from aphelion.backtest.order import Order, OrderType
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.clock import MarketClock
from aphelion.core.config import SENTINEL, Timeframe
from aphelion.core.data_layer import Bar, DataLayer
from aphelion.core.event_bus import EventBus
from aphelion.features.engine import FeatureEngine
from aphelion.intelligence.hydra.inference import HydraInference, HydraSignal
from aphelion.intelligence.hydra.strategy import HydraStrategy, StrategyConfig
from aphelion.paper.feed import DataFeed, FeedMode, ReplayFeed, SimulatedFeed, SimulatedFeedConfig
from aphelion.paper.ledger import PaperLedger
from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
from aphelion.risk.sentinel.core import SentinelCore
from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer
from aphelion.risk.sentinel.execution.paper import PaperConfig, PaperExecutor
from aphelion.risk.sentinel.monitor import SentinelMonitor
from aphelion.risk.sentinel.position_sizer import PositionSizer
from aphelion.risk.sentinel.validator import TradeValidator

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PaperSessionConfig:
    """Configuration for a paper trading session."""
    # Identity
    session_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

    # Financial
    initial_capital: float = 10_000.0
    symbol: str = "XAUUSD"

    # HYDRA
    hydra_checkpoint: str = ""             # Path to HydraGate checkpoint
    hydra_device: str = ""                 # "cuda" / "cpu" / "" (auto)

    # Strategy
    strategy_config: StrategyConfig = field(default_factory=StrategyConfig)

    # Paper executor
    paper_config: PaperConfig = field(default_factory=PaperConfig)

    # Warmup
    warmup_bars: int = 64                  # Need this many bars before trading

    # Sentinel status logging interval
    sentinel_log_interval_bars: int = 50   # Log status every N bars

    # Friday close
    friday_close_enabled: bool = True

    # Monitor (async SL breach checker)
    monitor_enabled: bool = True


# ── Session Runner ───────────────────────────────────────────────────────────

class PaperSession:
    """
    Async paper trading session.

    Usage:
        session = PaperSession(config, feed)
        results = await session.run()
    """

    def __init__(
        self,
        config: PaperSessionConfig,
        feed: DataFeed,
    ):
        self._config = config
        self._feed = feed

        # ── Wire the full SENTINEL stack ──────────────────────────────────
        self._event_bus = EventBus()
        self._clock = MarketClock()
        self._data_layer = DataLayer(self._event_bus)

        self._sentinel_core = SentinelCore(self._event_bus, self._clock)
        self._validator = TradeValidator(self._sentinel_core, self._clock)
        self._cb = CircuitBreaker(self._event_bus)
        self._enforcer = ExecutionEnforcer(self._validator, self._cb)
        self._sizer = PositionSizer()
        self._portfolio = Portfolio(config.initial_capital)

        self._paper_executor = PaperExecutor(
            config=config.paper_config,
            enforcer=self._enforcer,
            sentinel_core=self._sentinel_core,
            portfolio=self._portfolio,
            event_bus=self._event_bus,
        )

        self._monitor: Optional[SentinelMonitor] = None
        if config.monitor_enabled:
            self._monitor = SentinelMonitor(self._event_bus, self._sentinel_core)

        # ── Feature engine ────────────────────────────────────────────────
        self._feature_engine = FeatureEngine(self._data_layer, self._clock)

        # ── HYDRA inference + strategy ────────────────────────────────────
        self._inference: Optional[HydraInference] = None
        self._strategy: Optional[HydraStrategy] = None

        # ── Ledger ────────────────────────────────────────────────────────
        self._ledger = PaperLedger(config.session_id)

        # ── Runtime state ─────────────────────────────────────────────────
        self._bar_count: int = 0
        self._error_count: int = 0
        self._running: bool = False

    # ── Initialization ────────────────────────────────────────────────────

    def _init_hydra(self) -> None:
        """Load HYDRA model and create strategy adapter."""
        if self._config.hydra_checkpoint:
            try:
                self._inference = HydraInference(
                    checkpoint_path=self._config.hydra_checkpoint,
                    device=self._config.hydra_device or None,
                )
                self._strategy = HydraStrategy(
                    inference=self._inference,
                    config=self._config.strategy_config,
                )
                logger.info("HYDRA loaded from %s", self._config.hydra_checkpoint)
            except Exception as exc:
                logger.error("Failed to load HYDRA: %s", exc)
                self._ledger.log_error("HYDRA load failed", {"error": str(exc)})
                # Session can still run — it just won't generate signals
        else:
            logger.warning("No HYDRA checkpoint — session will not generate signals")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self) -> PaperSessionResult:
        """
        Run the paper trading session until the feed is exhausted or stopped.
        """
        self._running = True
        self._init_hydra()

        # Start SL monitor
        if self._monitor:
            await self._monitor.start()

        start_time = datetime.now(timezone.utc)
        self._ledger.log_event("RUN_START", {
            "config": {
                "initial_capital": self._config.initial_capital,
                "warmup_bars": self._config.warmup_bars,
                "hydra_checkpoint": self._config.hydra_checkpoint,
                "symbol": self._config.symbol,
            }
        })

        logger.info(
            "Paper session %s started — capital=%.0f, warmup=%d bars",
            self._config.session_id,
            self._config.initial_capital,
            self._config.warmup_bars,
        )

        try:
            async for bar in self._feed.bars():
                if not self._running:
                    break
                try:
                    self._process_bar(bar)
                except Exception as exc:
                    self._error_count += 1
                    logger.exception("Error processing bar %d", self._bar_count)
                    self._ledger.log_error(
                        f"Bar processing error at bar {self._bar_count}",
                        {"error": str(exc), "bar_count": self._bar_count},
                    )
        except Exception as fatal:
            self._error_count += 1
            logger.exception("Fatal feed error")
            self._ledger.log_error("Fatal feed error", {"error": str(fatal)})

        # Stop monitor
        if self._monitor:
            await self._monitor.stop()

        end_time = datetime.now(timezone.utc)

        # Build results
        result = self._build_result(start_time, end_time)

        # Write summary to ledger and close
        self._ledger.close(summary={
            "bars_processed": self._bar_count,
            "errors": self._error_count,
            "final_equity": self._portfolio.equity,
            "trades": len(self._portfolio.get_closed_trades()),
            "fills": self._paper_executor.stats["fill_count"],
            "rejections": self._paper_executor.stats["rejection_count"],
        })

        logger.info(
            "Paper session %s completed — %d bars, %d trades, equity=%.2f, errors=%d",
            self._config.session_id,
            self._bar_count,
            len(self._portfolio.get_closed_trades()),
            self._portfolio.equity,
            self._error_count,
        )

        return result

    # ── Per-bar processing ────────────────────────────────────────────────

    def _process_bar(self, bar: Bar) -> None:
        """Process a single bar through the full pipeline."""
        self._bar_count += 1

        # 1. Update simulated clock
        bar_ts = bar.timestamp if isinstance(bar.timestamp, datetime) else None
        if bar_ts:
            self._clock.set_simulated_time(bar_ts)

        # 2. Feed bar to DataLayer for feature computation
        tf = getattr(bar, "timeframe", Timeframe.M1)
        if tf in self._data_layer._bars:
            self._data_layer._bars[tf].append(bar)

        # 3. Update SENTINEL equity tracking
        self._sentinel_core.update_equity(self._portfolio.equity)
        self._cb.update(self._portfolio.equity)

        # 4. Handle L3 halt
        if self._sentinel_core.l3_triggered:
            closed = self._paper_executor.force_close_all(bar.close, "L3_HALT")
            if closed:
                self._ledger.log_event("L3_HALT", {
                    "positions_closed": closed,
                    "price": bar.close,
                    "equity": self._portfolio.equity,
                })
            return  # No further processing when halted

        # 5. Handle Friday close
        if self._config.friday_close_enabled:
            self._handle_friday_close(bar)

        # 6. Check SL/TP on open positions
        exits = self._paper_executor.check_sl_tp(bar.close)
        for position_id, exit_price, reason in exits:
            pnl = self._paper_executor.close_position(position_id, exit_price, reason)
            self._ledger.log_exit({
                "position_id": position_id,
                "exit_price": exit_price,
                "reason": reason,
                "net_pnl": pnl,
                "bar_count": self._bar_count,
            })

        # 7. Check pending orders
        fills = self._paper_executor.check_pending_orders(bar.close)
        for fill in fills:
            self._ledger.log_fill({
                "order_id": fill.order_id,
                "fill_price": fill.filled_price,
                "side": fill.side.value,
                "size_lots": fill.size_lots,
                "bar_count": self._bar_count,
            })

        # 8. Update monitor price
        if self._monitor:
            self._monitor.update_price(bar.close)

        # 9. Compute features
        features: dict = {}
        try:
            features = self._feature_engine.on_bar(bar)
        except Exception:
            logger.debug("Feature engine error on bar %d", self._bar_count, exc_info=True)

        # 10. Strategy evaluation (only after warmup)
        if self._bar_count >= self._config.warmup_bars and self._strategy is not None:
            try:
                orders = self._strategy(bar, features, self._portfolio)
                if orders:
                    for order in orders:
                        self._submit_order(order, bar)
            except Exception as exc:
                logger.debug("Strategy error on bar %d: %s", self._bar_count, exc)

        # 11. Update portfolio mark-to-market
        self._portfolio.update_bar(bar, self._bar_count)
        self._paper_executor.update_price(bar.close, self._bar_count)

        # 12. Periodic sentinel status logging
        if self._bar_count % self._config.sentinel_log_interval_bars == 0:
            status = self._sentinel_core.get_status()
            status["bar_count"] = self._bar_count
            status["portfolio_equity"] = self._portfolio.equity
            self._ledger.log_sentinel_status(status)

    # ── Order submission ──────────────────────────────────────────────────

    def _submit_order(self, order: Order, bar: Bar) -> None:
        """Submit an order through the paper executor with full logging."""
        fill = self._paper_executor.submit_order(order, bar.close)
        if fill:
            self._ledger.log_fill({
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "fill_price": fill.filled_price,
                "size_lots": fill.size_lots,
                "commission": fill.commission,
                "slippage": fill.slippage,
                "bar_count": self._bar_count,
            })
        else:
            # Rejection or pending (pending fills are logged in check_pending_orders)
            if order.status.value == "REJECTED":
                self._ledger.log_rejection({
                    "order_id": order.order_id,
                    "proposed_by": order.proposed_by,
                    "bar_count": self._bar_count,
                })

    # ── Friday close ──────────────────────────────────────────────────────

    def _handle_friday_close(self, bar: Bar) -> None:
        """Force-close all positions 30 min before Friday market close."""
        ts = bar.timestamp if isinstance(bar.timestamp, datetime) else None
        if ts is None or ts.weekday() != 4:
            return

        close_time = ts.replace(hour=21, minute=0, second=0, microsecond=0)
        lockout_start = close_time - timedelta(minutes=SENTINEL.friday_close_lockout_minutes)

        if ts >= lockout_start:
            closed = self._paper_executor.force_close_all(bar.close, "FRIDAY_CLOSE")
            if closed:
                self._ledger.log_event("FRIDAY_CLOSE", {
                    "positions_closed": closed,
                    "price": bar.close,
                    "bar_count": self._bar_count,
                })

    # ── Stop ──────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the session to stop after the current bar."""
        self._running = False
        self._feed.stop()

    # ── Result builder ────────────────────────────────────────────────────

    def _build_result(self, start: datetime, end: datetime) -> "PaperSessionResult":
        """Compile session results."""
        trades = self._portfolio.get_closed_trades()
        eq_ts, eq_vals = self._portfolio.get_equity_series()
        dd_ts, dd_vals = self._portfolio.get_drawdown_series()

        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]

        return PaperSessionResult(
            session_id=self._config.session_id,
            start_time=start,
            end_time=end,
            bars_processed=self._bar_count,
            error_count=self._error_count,
            initial_capital=self._config.initial_capital,
            final_equity=self._portfolio.equity,
            peak_equity=self._portfolio.peak_equity,
            max_drawdown=max(dd_vals) if dd_vals else 0.0,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0.0,
            total_pnl=sum(t.net_pnl for t in trades),
            avg_r_multiple=sum(t.r_multiple for t in trades) / len(trades) if trades else 0.0,
            fill_count=self._paper_executor.stats["fill_count"],
            rejection_count=self._paper_executor.stats["rejection_count"],
            enforcer_stats=self._enforcer.stats,
            sentinel_status=self._sentinel_core.get_status(),
            ledger_path=str(self._ledger.path),
            equity_curve=(eq_ts, eq_vals),
            drawdown_curve=(dd_ts, dd_vals),
        )


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class PaperSessionResult:
    """Comprehensive results from a paper trading session."""
    session_id: str
    start_time: datetime
    end_time: datetime
    bars_processed: int
    error_count: int

    # Financial
    initial_capital: float
    final_equity: float
    peak_equity: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_r_multiple: float

    # Execution
    fill_count: int
    rejection_count: int
    enforcer_stats: dict
    sentinel_status: dict
    ledger_path: str

    # Curves
    equity_curve: tuple
    drawdown_curve: tuple

    @property
    def return_pct(self) -> float:
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    def summary(self) -> str:
        """Human-readable session summary."""
        return (
            f"═══ Paper Session {self.session_id} ═══\n"
            f"Duration:      {self.duration}\n"
            f"Bars:          {self.bars_processed}\n"
            f"Errors:        {self.error_count}\n"
            f"─── Financial ───\n"
            f"Initial:       ${self.initial_capital:,.2f}\n"
            f"Final:         ${self.final_equity:,.2f}\n"
            f"Return:        {self.return_pct:+.2f}%\n"
            f"Max DD:        {self.max_drawdown:.2%}\n"
            f"─── Trades ───\n"
            f"Total:         {self.total_trades}\n"
            f"Win/Loss:      {self.winning_trades}/{self.losing_trades}\n"
            f"Win Rate:      {self.win_rate:.1%}\n"
            f"Avg R:         {self.avg_r_multiple:.2f}\n"
            f"─── Execution ───\n"
            f"Fills:         {self.fill_count}\n"
            f"Rejections:    {self.rejection_count}\n"
            f"─── SENTINEL ───\n"
            f"L3 Triggered:  {self.sentinel_status.get('l3_triggered', False)}\n"
            f"Trading OK:    {self.sentinel_status.get('trading_allowed', '?')}\n"
            f"═══════════════════════\n"
            f"Ledger: {self.ledger_path}\n"
        )
