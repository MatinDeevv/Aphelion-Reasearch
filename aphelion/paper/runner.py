"""
APHELION Paper Trading Runner — Phase 5
End-to-end orchestrator that wires:
    MT5TickFeed → EventBus → FeatureEngine → HydraStrategy
    → PaperSession → SENTINEL → PaperExecutor → TUI → Ledger

Usage:
    runner = PaperRunner(config)
    result = await runner.run()
"""

from __future__ import annotations

import asyncio
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from aphelion.core.clock import MarketClock
from aphelion.core.config import EventTopic, Timeframe
from aphelion.core.data_layer import Bar, DataLayer, Tick
from aphelion.core.event_bus import Event, EventBus, Priority
from aphelion.paper.feed import (
    DataFeed,
    FeedConfig,
    FeedMode,
    MT5TickFeed,
    ReplayFeed,
    SimulatedFeed,
    SimulatedFeedConfig,
)
from aphelion.paper.session import PaperSession, PaperSessionConfig, PaperSessionResult
from aphelion.risk.sentinel.execution.mt5 import MT5Config, MT5Connection
from aphelion.tui.bridge import TUIBridge
from aphelion.tui.state import TUIState, PositionView

logger = logging.getLogger(__name__)

# Optional imports for system stats
try:
    import psutil  # type: ignore[import-untyped]
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PaperRunnerConfig:
    """Full configuration for an end-to-end paper trading run."""

    # Feed mode
    feed_mode: FeedMode = FeedMode.SIMULATED

    # MT5 connection (only used when feed_mode == MT5_TICK or LIVE)
    mt5_config: MT5Config = field(default_factory=MT5Config)

    # Tick feed config (FeedMode.MT5_TICK)
    feed_config: FeedConfig = field(default_factory=FeedConfig)

    # Session config (financial, HYDRA, strategy, executor)
    session_config: PaperSessionConfig = field(default_factory=PaperSessionConfig)

    # Simulated feed config (FeedMode.SIMULATED)
    sim_config: SimulatedFeedConfig = field(default_factory=SimulatedFeedConfig)

    # Replay bar list (FeedMode.REPLAY — caller supplies)
    replay_bars: list[Bar] = field(default_factory=list)

    # TUI
    enable_tui: bool = True
    tui_refresh_interval_s: float = 0.25

    # System stats polling
    system_stats_interval_s: float = 5.0


# ── Runner ───────────────────────────────────────────────────────────────────

class PaperRunner:
    """
    End-to-end paper trading runner.

    Orchestrates:
    1. EventBus + MarketClock setup
    2. MT5TickFeed (or SimulatedFeed/ReplayFeed) startup
    3. TUI bridge wiring
    4. PaperSession execution
    5. Graceful shutdown and result collection
    """

    def __init__(self, config: Optional[PaperRunnerConfig] = None):
        self._config = config or PaperRunnerConfig()

        # Core components (created on run)
        self._event_bus: Optional[EventBus] = None
        self._clock: Optional[MarketClock] = None
        self._mt5_conn: Optional[MT5Connection] = None
        self._tick_feed: Optional[MT5TickFeed] = None
        self._feed: Optional[DataFeed] = None
        self._session: Optional[PaperSession] = None
        self._tui_state: Optional[TUIState] = None
        self._tui_bridge: Optional[TUIBridge] = None

        # Runtime
        self._running: bool = False
        self._start_time: float = 0.0
        self._stats_task: Optional[asyncio.Task] = None
        self._session_sync_task: Optional[asyncio.Task] = None

    @property
    def tui_state(self) -> Optional[TUIState]:
        return self._tui_state

    # ── Public API ────────────────────────────────────────────────────────

    async def run(self) -> PaperSessionResult:
        """
        Run the paper trading session end-to-end.
        Returns PaperSessionResult when complete.
        """
        self._running = True
        self._start_time = time.monotonic()

        # 1. Create shared infrastructure
        self._event_bus = EventBus()
        self._clock = MarketClock()
        self._clock.auto_detect_dst()

        # 2. Create TUI state + bridge
        if self._config.enable_tui:
            self._tui_state = TUIState(
                session_name=self._config.session_config.session_id,
                session_start=datetime.now(timezone.utc),
            )
            self._tui_bridge = TUIBridge(self._tui_state)
            self._subscribe_events()

        # 3. Create data feed based on mode
        feed = await self._create_feed()

        # 4. Create and run paper session
        self._session = PaperSession(self._config.session_config, feed)

        # Start system stats poller
        if self._config.enable_tui:
            self._stats_task = asyncio.create_task(self._system_stats_loop())
            self._session_sync_task = asyncio.create_task(self._session_state_loop())

        logger.info(
            "PaperRunner starting — mode=%s, capital=%.0f, session=%s",
            self._config.feed_mode.name,
            self._config.session_config.initial_capital,
            self._config.session_config.session_id,
        )

        try:
            result = await self._session.run()
        finally:
            await self._shutdown()

        logger.info("PaperRunner complete — %s", result.session_id)
        return result

    async def stop(self) -> None:
        """Signal graceful stop."""
        self._running = False
        if self._session:
            self._session.stop()

    # ── Feed creation ─────────────────────────────────────────────────────

    async def _create_feed(self) -> DataFeed:
        """Create the appropriate data feed based on config."""
        mode = self._config.feed_mode

        if mode == FeedMode.MT5_TICK:
            return await self._create_mt5_tick_feed()
        elif mode == FeedMode.LIVE:
            return await self._create_mt5_bar_feed()
        elif mode == FeedMode.REPLAY:
            self._feed = ReplayFeed(self._config.replay_bars)
            return self._feed
        else:
            # Default: SIMULATED
            self._feed = SimulatedFeed(self._config.sim_config)
            return self._feed

    async def _create_mt5_tick_feed(self) -> MT5TickFeed:
        """Create and start the MT5 tick-level feed."""
        self._mt5_conn = MT5Connection(self._config.mt5_config)
        self._tick_feed = MT5TickFeed(
            mt5_connection=self._mt5_conn,
            event_bus=self._event_bus,
            config=self._config.feed_config,
        )

        ok = await self._tick_feed.start()
        if not ok:
            raise ConnectionError(
                "Failed to connect to MT5. Check: terminal is running and logged in, "
                "allow algo/API access in MT5 settings, credentials/server are correct, "
                "and MT5 terminal bitness matches Python (typically 64-bit)."
            )

        self._feed = self._tick_feed
        return self._tick_feed

    async def _create_mt5_bar_feed(self) -> DataFeed:
        """Create the bar-level MT5 feed (LiveMT5Feed)."""
        from aphelion.paper.feed import LiveMT5Feed

        self._mt5_conn = MT5Connection(self._config.mt5_config)
        ok = self._mt5_conn.connect()
        if not ok:
            raise ConnectionError("Failed to connect to MT5 for bar-level feed.")

        feed = LiveMT5Feed(self._mt5_conn)
        self._feed = feed
        return feed

    # ── Event subscriptions ───────────────────────────────────────────────

    def _subscribe_events(self) -> None:
        """Wire EventBus topics to TUI bridge updates."""
        if not self._event_bus or not self._tui_bridge:
            return

        self._event_bus.subscribe(EventTopic.TICK, self._on_tick)
        self._event_bus.subscribe(EventTopic.BAR, self._on_bar_event)
        self._event_bus.subscribe(EventTopic.SYSTEM, self._on_system_event)

    async def _on_tick(self, event: Event) -> None:
        """Handle tick events — update TUI price display."""
        if not self._tui_bridge:
            return
        tick = event.data
        if isinstance(tick, Tick):
            self._tui_bridge.update_price(
                bid=tick.bid,
                ask=tick.ask,
            )

    async def _on_bar_event(self, event: Event) -> None:
        """Handle bar events — update TUI bar counter and session info."""
        if not self._tui_bridge or not self._clock:
            return

        data = event.data
        if isinstance(data, dict) and "bar" in data:
            bar = data["bar"]
            if isinstance(bar, Bar):
                bar_ts = bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now(timezone.utc)
                session = self._clock.current_session(bar_ts)
                self._tui_bridge.update_bar(
                    bar_time=bar_ts,
                    session_name=session.name,
                    market_open=self._clock.is_market_open(bar_ts),
                    bars_processed=self._tui_state.bars_processed + 1 if self._tui_state else 0,
                )

    async def _on_system_event(self, event: Event) -> None:
        """Handle system events (feed status changes)."""
        if not self._tui_bridge:
            return
        data = event.data
        if isinstance(data, dict):
            event_type = data.get("type", "")
            if event_type.startswith("feed."):
                self._tui_bridge.log("SYSTEM", f"Feed event: {event_type}")
                if event_type in ("feed.error", "feed.disconnected"):
                    self._tui_bridge.alert(
                        "WARNING", "Feed Issue", f"{event_type}: {data}"
                    )

    # ── System stats ──────────────────────────────────────────────────────

    async def _system_stats_loop(self) -> None:
        """Periodically push system stats to TUI."""
        while self._running:
            try:
                if self._tui_bridge:
                    cpu = 0.0
                    mem = 0.0
                    if HAS_PSUTIL:
                        cpu = psutil.cpu_percent(interval=0)
                        mem = psutil.Process().memory_info().rss / (1024 * 1024)

                    uptime = time.monotonic() - self._start_time
                    latency = 0.0
                    if self._tick_feed:
                        stats = self._tick_feed.get_stats()
                        latency = (time.monotonic() - stats.last_tick_time) * 1000 if stats.last_tick_time > 0 else 0.0

                    self._tui_bridge.update_system_stats(
                        cpu_usage=cpu,
                        memory_mb=mem,
                        latency_ms=latency,
                        uptime_seconds=uptime,
                    )
            except Exception:
                pass

            await asyncio.sleep(self._config.system_stats_interval_s)

    async def _session_state_loop(self) -> None:
        """Mirror PaperSession runtime state into TUI in real time."""
        while self._running:
            try:
                if self._tui_bridge and self._session and self._tui_state:
                    portfolio = getattr(self._session, "_portfolio", None)
                    sentinel = getattr(self._session, "_sentinel_core", None)
                    executor = getattr(self._session, "_paper_executor", None)
                    bar_count = getattr(self._session, "_bar_count", 0)
                    clock = getattr(self._session, "_clock", self._clock)

                    if portfolio is not None:
                        trades = portfolio.get_closed_trades()
                        wins = sum(1 for t in trades if t.net_pnl > 0)
                        losses = len(trades) - wins
                        realized = sum(t.net_pnl for t in trades)
                        unrealized = portfolio.equity - portfolio.cash
                        daily_pnl = portfolio.equity - self._config.session_config.initial_capital

                        self._tui_bridge.update_equity(
                            account_equity=portfolio.equity,
                            daily_pnl=daily_pnl,
                            realized_pnl=realized,
                            unrealized_pnl=unrealized,
                            total_trades=len(trades),
                            winning_trades=wins,
                            losing_trades=losses,
                        )

                        open_positions = []
                        for position in portfolio._open_positions.values():
                            open_positions.append(
                                PositionView(
                                    position_id=position.position_id,
                                    symbol=position.symbol,
                                    direction=position.direction,
                                    entry_price=position.entry_price,
                                    current_price=position.entry_price,
                                    stop_loss=position.stop_loss,
                                    take_profit=position.take_profit,
                                    size_lots=position.size_lots,
                                    unrealized_pnl=position.unrealized_pnl,
                                    open_time=position.open_time,
                                )
                            )
                        self._tui_bridge.update_positions(open_positions)

                    if sentinel is not None:
                        status = sentinel.get_status()
                        self._tui_bridge.update_sentinel(
                            l1=bool(status.get("l1_triggered", False)),
                            l2=bool(status.get("l2_triggered", False)),
                            l3=bool(status.get("l3_triggered", False)),
                            open_positions=int(status.get("open_positions", 0)),
                            total_exposure_pct=float(status.get("total_exposure_pct", 0.0)),
                            daily_drawdown_pct=float(status.get("daily_drawdown", 0.0)),
                            session_peak_equity=float(status.get("session_peak_equity", 0.0)),
                            trading_allowed=bool(status.get("trading_allowed", True)),
                        )

                    if clock is not None:
                        now = clock.now_utc()
                        self._tui_bridge.update_bar(
                            bar_time=now,
                            session_name=clock.current_session(now).name,
                            market_open=clock.is_market_open(now),
                            bars_processed=bar_count,
                        )

                    if self._tick_feed:
                        stats = self._tick_feed.get_stats()
                        self._tui_bridge.update_feed_status(
                            connected=self._tick_feed.is_connected,
                            mode=self._config.feed_mode.name,
                            ticks_per_min=stats.ticks_per_minute,
                            reconnect_count=stats.reconnect_count,
                        )
                    else:
                        self._tui_bridge.update_feed_status(
                            connected=True,
                            mode=self._config.feed_mode.name,
                            ticks_per_min=0.0,
                            reconnect_count=0,
                        )

                    self._tui_bridge.update_session_duration(
                        (time.monotonic() - self._start_time) / 60.0
                    )

                    if executor is not None:
                        self._tui_bridge.update_sentinel_rejections(
                            int(executor.stats.get("rejection_count", 0))
                        )
            except Exception:
                logger.debug("Session state sync error", exc_info=True)

            await asyncio.sleep(0.25)

    # ── Shutdown ──────────────────────────────────────────────────────────

    async def _shutdown(self) -> None:
        """Clean up all resources."""
        self._running = False

        # Cancel stats polling
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass

        if self._session_sync_task and not self._session_sync_task.done():
            self._session_sync_task.cancel()
            try:
                await self._session_sync_task
            except asyncio.CancelledError:
                pass

        # Stop tick feed
        if self._tick_feed:
            await self._tick_feed.stop()

        # Disconnect MT5
        if self._mt5_conn and self._mt5_conn.is_connected:
            self._mt5_conn.disconnect()

        logger.info("PaperRunner shutdown complete")
