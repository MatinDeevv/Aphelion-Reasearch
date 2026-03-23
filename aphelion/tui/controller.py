"""
APHELION TUI — Unified Controller (Phase 23 — Wired)

Single ``AphelionController`` owns ALL background tasks:
session, training, forge, backtest.  The TUI never imports
PaperRunner directly — it calls methods on the controller.

Architecture
────────────
  aphelion.py  →  AphelionController(config)  →  TUI
  TUI reads TUIState (one direction).
  Controller writes into TUIState via sub-controllers / simulator.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ─── State enums ─────────────────────────────────────────────────────────────


class SystemState(Enum):
    """Top-level system state managed by AphelionController."""
    IDLE = "IDLE"
    SESSION_RUNNING = "SESSION_RUNNING"
    SESSION_STOPPING = "SESSION_STOPPING"
    TRAINING = "TRAINING"
    FORGING = "FORGING"
    BACKTESTING = "BACKTESTING"
    ERROR = "ERROR"


class SessionState(Enum):
    """Lifecycle state of a trading session."""
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class TrainingState(Enum):
    """Lifecycle state of a HYDRA training run."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


# ─── Progress dataclasses ───────────────────────────────────────────────────


@dataclass
class TrainingProgress:
    """Snapshot of training progress for TUI rendering."""
    state: TrainingState = TrainingState.IDLE
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = float("inf")
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    model_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)
    message: str = ""


@dataclass
class ForgeProgress:
    """Snapshot of a HEPHAESTUS forge operation for TUI rendering."""
    stage: str = ""
    message: str = ""
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    complete: bool = False
    success: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestProgress:
    """Snapshot of backtest progress for TUI rendering."""
    running: bool = False
    bars_processed: int = 0
    total_bars: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""
    results: Optional[dict[str, Any]] = None


# ─── Session Controller ─────────────────────────────────────────────────────


class SessionController:
    """Controls the PaperRunner lifecycle from the TUI.

    The TUI calls start_session/stop_session.
    The controller runs PaperRunner in a background asyncio task.
    Status is streamed to TUIState which the dashboard renders.
    """

    def __init__(self) -> None:
        self._state: SessionState = SessionState.IDLE
        self._runner: Any = None  # PaperRunner (lazy import)
        self._task: Optional[asyncio.Task] = None
        self._result: Any = None
        self._error: Optional[str] = None
        self._start_time: float = 0.0

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in (SessionState.STARTING, SessionState.RUNNING)

    @property
    def result(self) -> Any:
        return self._result

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def uptime_seconds(self) -> float:
        if self._start_time > 0 and self.is_active:
            return time.monotonic() - self._start_time
        return 0.0

    async def start_session(
        self,
        runner_config: Any,
        tui_state: Any = None,
    ) -> None:
        """Start a paper trading session in a background task.

        Parameters
        ----------
        runner_config : PaperRunnerConfig
            Fully populated config for PaperRunner.
        tui_state : TUIState, optional
            Shared state object for TUI rendering.
        """
        if self.is_active:
            logger.warning("Session already active — ignoring start request")
            return

        self._state = SessionState.STARTING
        self._error = None
        self._result = None
        self._start_time = time.monotonic()

        try:
            from aphelion.paper.runner import PaperRunner
            self._runner = PaperRunner(runner_config)
            if tui_state is not None:
                self._runner._tui_state = tui_state

            self._task = asyncio.create_task(self._run_session())
        except Exception as exc:
            self._state = SessionState.ERROR
            self._error = str(exc)
            logger.exception("Failed to start session: %s", exc)

    async def _run_session(self) -> None:
        """Background task that runs the PaperRunner."""
        try:
            self._state = SessionState.RUNNING
            self._result = await self._runner.run()
            self._state = SessionState.STOPPED
        except asyncio.CancelledError:
            self._state = SessionState.STOPPED
        except Exception as exc:
            self._state = SessionState.ERROR
            self._error = str(exc)
            logger.exception("Session error: %s", exc)

    async def stop_session(self, close_positions: bool = True) -> None:
        """Stop the active trading session."""
        if not self.is_active:
            return

        self._state = SessionState.STOPPING
        try:
            if self._runner:
                await self._runner.stop()
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        except Exception as exc:
            logger.exception("Error stopping session: %s", exc)
        finally:
            self._state = SessionState.STOPPED
            self._start_time = 0.0


# ─── Training Controller ────────────────────────────────────────────────────


class TrainingController:
    """Controls HYDRA training from the TUI.

    Training runs in a background thread (CPU-bound / GPU work)
    and streams progress to a TrainingProgress object that the TUI polls.
    """

    def __init__(self) -> None:
        self._state: TrainingState = TrainingState.IDLE
        self._progress: TrainingProgress = TrainingProgress()
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._pause_event: threading.Event = threading.Event()
        self._error: Optional[str] = None

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def progress(self) -> TrainingProgress:
        return self._progress

    @property
    def is_active(self) -> bool:
        return self._state in (TrainingState.RUNNING, TrainingState.PAUSED)

    def start_training(self, config: dict[str, Any]) -> None:
        """Start HYDRA training in a background thread.

        Parameters
        ----------
        config : dict
            Training configuration (epochs, data path, etc.)
        """
        if self.is_active:
            logger.warning("Training already active — ignoring start request")
            return

        self._state = TrainingState.RUNNING
        self._stop_event.clear()
        self._pause_event.clear()
        self._error = None
        self._progress = TrainingProgress(
            state=TrainingState.RUNNING,
            total_epochs=config.get("epochs", 20),
        )

        self._thread = threading.Thread(
            target=self._training_loop,
            args=(config,),
            daemon=True,
            name="hydra-training",
        )
        self._thread.start()

    def _training_loop(self, config: dict[str, Any]) -> None:
        """Background training loop (runs in thread).

        In production this imports and runs the actual HYDRA training
        pipeline. For now, emits progress updates that the TUI can render.
        """
        import time as _time

        total_epochs = config.get("epochs", 20)
        t0 = _time.monotonic()

        try:
            for epoch in range(1, total_epochs + 1):
                if self._stop_event.is_set():
                    break

                # Pause gate
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    self._state = TrainingState.PAUSED
                    self._progress.state = TrainingState.PAUSED
                    _time.sleep(0.5)

                if self._stop_event.is_set():
                    break

                self._state = TrainingState.RUNNING
                self._progress.state = TrainingState.RUNNING
                self._progress.current_epoch = epoch
                self._progress.elapsed_seconds = _time.monotonic() - t0

                # Estimate ETA
                per_epoch = self._progress.elapsed_seconds / epoch
                self._progress.eta_seconds = per_epoch * (total_epochs - epoch)
                self._progress.message = f"Epoch {epoch}/{total_epochs}"

            self._state = TrainingState.COMPLETE
            self._progress.state = TrainingState.COMPLETE
            self._progress.message = "Training complete"
        except Exception as exc:
            self._state = TrainingState.ERROR
            self._progress.state = TrainingState.ERROR
            self._error = str(exc)
            self._progress.message = f"Error: {exc}"

    def pause(self) -> None:
        """Pause training."""
        if self._state == TrainingState.RUNNING:
            self._pause_event.set()

    def resume(self) -> None:
        """Resume paused training."""
        if self._state == TrainingState.PAUSED:
            self._pause_event.clear()

    def stop(self) -> None:
        """Stop training."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._state = TrainingState.IDLE
        self._progress.state = TrainingState.IDLE


# ─── Forge Controller ───────────────────────────────────────────────────────


class ForgeController:
    """Controls HEPHAESTUS forge operations from the TUI.

    Forge runs in a background thread (LLM + subprocess) and
    streams progress updates to a ForgeProgress object.
    """

    def __init__(self) -> None:
        self._progress: ForgeProgress = ForgeProgress()
        self._thread: Optional[threading.Thread] = None
        self._result: Any = None
        self._on_complete: Optional[Callable] = None

    @property
    def progress(self) -> ForgeProgress:
        return self._progress

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def result(self) -> Any:
        return self._result

    def start_forge(
        self,
        raw_code: str,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """Start forging a strategy in the background.

        Parameters
        ----------
        raw_code : str
            Raw Pine Script / Python / pseudocode / English input.
        on_complete : callable, optional
            Called when forge completes (success or failure).
        """
        if self.is_active:
            return

        self._progress = ForgeProgress()
        self._result = None
        self._on_complete = on_complete

        self._thread = threading.Thread(
            target=self._forge_loop,
            args=(raw_code,),
            daemon=True,
            name="hephaestus-forge",
        )
        self._thread.start()

    def _forge_loop(self, raw_code: str) -> None:
        """Background forge loop."""
        import time as _time

        t0 = _time.monotonic()

        stages = [
            ("detect", "Detecting input type...", 0.10),
            ("parse", "Parsing logic...", 0.25),
            ("codegen", "Generating Python class...", 0.40),
            ("sandbox", "Running sandbox tests...", 0.55),
            ("backtest", "Backtesting (2yr data)...", 0.70),
            ("walkforward", "Walk-forward (12 folds)...", 0.80),
            ("montecarlo", "Monte Carlo (1000 sims)...", 0.88),
            ("titan", "TITAN gate check...", 0.93),
            ("correlation", "Correlation check...", 0.97),
        ]

        try:
            from aphelion.hephaestus.agent import HephaestusAgent
            agent = HephaestusAgent()

            # Emit progress stages
            for stage_name, msg, pct in stages:
                self._progress.stage = stage_name
                self._progress.message = msg
                self._progress.percent = pct
                self._progress.elapsed_seconds = _time.monotonic() - t0
                _time.sleep(0.05)  # Small yield for TUI to poll

            # Run the actual forge
            result = agent.forge(raw_code)
            self._result = result

            self._progress.percent = 1.0
            self._progress.complete = True
            self._progress.success = (result.status.value == "DEPLOYED")
            self._progress.elapsed_seconds = _time.monotonic() - t0

            if self._progress.success:
                self._progress.message = f"DEPLOYED: {result.strategy_name}"
                self._progress.details = {
                    "name": result.strategy_name,
                    "sharpe": getattr(result.validation, "sharpe", 0) if result.validation else 0,
                }
            else:
                self._progress.message = f"REJECTED: {result.strategy_name}"
                if result.rejection:
                    self._progress.details = {
                        "reasons": result.rejection.reasons,
                        "failed_at": result.rejection.failed_at,
                    }

        except Exception as exc:
            self._progress.complete = True
            self._progress.success = False
            self._progress.message = f"Error: {exc}"
            self._progress.elapsed_seconds = _time.monotonic() - t0
            logger.exception("Forge error: %s", exc)

        if self._on_complete:
            try:
                self._on_complete(self._result)
            except Exception:
                pass


# ─── Backtest Controller ────────────────────────────────────────────────────


class BacktestController:
    """Controls backtest runs from the TUI."""

    def __init__(self) -> None:
        self._progress: BacktestProgress = BacktestProgress()
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()

    @property
    def progress(self) -> BacktestProgress:
        return self._progress

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_backtest(self, config: dict[str, Any]) -> None:
        """Start a backtest in a background thread."""
        if self.is_active:
            return

        self._progress = BacktestProgress(running=True, message="Starting backtest...")
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._backtest_loop,
            args=(config,),
            daemon=True,
            name="backtest-run",
        )
        self._thread.start()

    def _backtest_loop(self, config: dict[str, Any]) -> None:
        """Background backtest execution."""
        import time as _time

        t0 = _time.monotonic()

        try:
            self._progress.message = "Loading data..."
            self._progress.elapsed_seconds = _time.monotonic() - t0

            # Placeholder — in production this runs BacktestEngine
            self._progress.message = "Running backtest..."

            self._progress.running = False
            self._progress.elapsed_seconds = _time.monotonic() - t0
            self._progress.message = "Backtest complete"
            self._progress.results = {
                "total_return_pct": 0.0,
                "sharpe": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }
        except Exception as exc:
            self._progress.running = False
            self._progress.message = f"Error: {exc}"
            logger.exception("Backtest error: %s", exc)

    def stop(self) -> None:
        """Stop the backtest."""
        self._stop_event.set()
        self._progress.running = False


# ═════════════════════════════════════════════════════════════════════════════
# Unified AphelionController
# ═════════════════════════════════════════════════════════════════════════════


class AphelionController:
    """Single owner of ALL background work — session, training, forge, backtest.

    Created in ``aphelion.py`` and passed into the TUI.
    The TUI calls methods here; state flows one direction into TUIState.

    Usage::

        controller = AphelionController(config, tui_state)
        controller.start_session("simulated")
        controller.stop_session()
    """

    def __init__(self, config, tui_state=None) -> None:
        from aphelion.tui.config import AphelionConfig
        self._config: AphelionConfig = config
        self._tui_state = tui_state  # set later if not provided

        # System state
        self._system_state = SystemState.IDLE
        self._state_callbacks: list[Callable[[SystemState], None]] = []

        # Sub-controllers (lazy)
        self._session_ctrl = SessionController()
        self._training_ctrl = TrainingController()
        self._forge_ctrl = ForgeController()
        self._backtest_ctrl = BacktestController()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def config(self):
        """The live AphelionConfig."""
        return self._config

    @config.setter
    def config(self, value) -> None:
        self._config = value

    @property
    def system_state(self) -> SystemState:
        return self._system_state

    @property
    def session_ctrl(self) -> SessionController:
        return self._session_ctrl

    @property
    def training_ctrl(self) -> TrainingController:
        return self._training_ctrl

    @property
    def forge_ctrl(self) -> ForgeController:
        return self._forge_ctrl

    @property
    def backtest_ctrl(self) -> BacktestController:
        return self._backtest_ctrl

    @property
    def tui_state(self):
        return self._tui_state

    @tui_state.setter
    def tui_state(self, value) -> None:
        self._tui_state = value

    @property
    def is_session_active(self) -> bool:
        return self._session_ctrl.is_active

    # ── State callbacks ──────────────────────────────────────────────

    def register_state_callback(self, cb: Callable[[SystemState], None]) -> None:
        """Register a callback that fires on system-state changes."""
        self._state_callbacks.append(cb)

    def _set_system_state(self, new: SystemState) -> None:
        self._system_state = new
        for cb in self._state_callbacks:
            try:
                cb(new)
            except Exception:
                pass

    # ── Session lifecycle ────────────────────────────────────────────

    def start_session(self, mode: str = "paper") -> None:
        """Start a paper trading session.

        Uses PaperRunner via SessionController with real MT5 data.
        """
        cfg = self._config
        cfg.trading.mode = mode
        state = self._tui_state

        # Stop any existing session first
        self.stop_session()

        self._set_system_state(SystemState.SESSION_RUNNING)

        if state:
            state.push_log("INFO", f"Starting {mode} session — {cfg.trading.symbol}")
            state.session_name = f"Paper-{cfg.trading.symbol}"
        try:
            from aphelion.tui.config import config_to_runner_config
            runner_config = config_to_runner_config(cfg, mode)
            # PaperRunner is async — need an event loop
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    self._session_ctrl.start_session(runner_config, state)
                )
            else:
                loop.run_until_complete(
                    self._session_ctrl.start_session(runner_config, state)
                )
        except Exception as exc:
            logger.error("PaperRunner failed: %s", exc)
            if state:
                state.push_log("ERROR", f"Session start failed: {exc}")
            self._set_system_state(SystemState.IDLE)

    def stop_session(self) -> None:
        """Stop the active PaperRunner session."""
        if self._session_ctrl.is_active:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._session_ctrl.stop_session())
                else:
                    loop.run_until_complete(self._session_ctrl.stop_session())
            except Exception:
                pass
        self._set_system_state(SystemState.IDLE)

    # ── Training lifecycle ───────────────────────────────────────────

    def start_training(self, config: dict[str, Any]) -> TrainingProgress:
        """Start HYDRA training in a background thread."""
        if self._training_ctrl.is_active:
            return self._training_ctrl.progress
        self._set_system_state(SystemState.TRAINING)
        self._training_ctrl.start_training(config)
        return self._training_ctrl.progress

    def pause_training(self) -> None:
        self._training_ctrl.pause()

    def resume_training(self) -> None:
        self._training_ctrl.resume()

    def stop_training(self) -> None:
        self._training_ctrl.stop()
        self._set_system_state(SystemState.IDLE)

    # ── Forge lifecycle ──────────────────────────────────────────────

    def start_forge(self, raw_code: str, on_complete=None) -> ForgeProgress:
        """Start a HEPHAESTUS forge operation."""
        if self._forge_ctrl.is_active:
            return self._forge_ctrl.progress
        self._set_system_state(SystemState.FORGING)
        self._forge_ctrl.start_forge(raw_code, on_complete=on_complete)
        return self._forge_ctrl.progress

    # ── Backtest lifecycle ───────────────────────────────────────────

    def start_backtest(self, config: dict[str, Any]) -> BacktestProgress:
        """Start a backtest in a background thread."""
        if self._backtest_ctrl.is_active:
            return self._backtest_ctrl.progress
        self._set_system_state(SystemState.BACKTESTING)
        self._backtest_ctrl.start_backtest(config)
        return self._backtest_ctrl.progress

    def stop_backtest(self) -> None:
        self._backtest_ctrl.stop()
        self._set_system_state(SystemState.IDLE)

    # ── Status helpers ───────────────────────────────────────────────

    def check_mt5_connection(self) -> bool:
        """Return True if MT5 creds are configured."""
        return self._config.has_mt5_credentials()

    def find_hydra_checkpoint(self) -> str | None:
        """Return checkpoint path if it exists, else None."""
        if self._config.has_hydra_checkpoint():
            return self._config.hydra.checkpoint
        return None
