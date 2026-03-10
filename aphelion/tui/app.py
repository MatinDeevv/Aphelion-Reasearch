"""
APHELION TUI — Core Application  (v2 — Bloomberg-grade)

Dual-mode TUI:
  • **Textual mode** (default when textual is installed):
    Full interactive terminal app with keyboard navigation, multi-view tabs,
    auto-refresh, and Bloomberg-style dark theme.

  • **Rich-Live fallback** (when textual is unavailable):
    Read-only Rich Live dashboard with 2 Hz refresh.

Both modes read from the same TUIState object that the PaperSession populates
via TUIBridge.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, Static, RichLog, TabbedContent, TabPane
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

from aphelion.tui.screens.dashboard import build_dashboard_layout, _build_status_bar
from aphelion.tui.screens.header import build_header
from aphelion.tui.screens.hydra_panel import build_hydra_panel
from aphelion.tui.screens.sentinel_panel import build_sentinel_panel
from aphelion.tui.screens.positions import build_positions_panel
from aphelion.tui.screens.equity import build_equity_panel
from aphelion.tui.screens.event_log import build_log_panel
from aphelion.tui.screens.performance import build_performance_panel
from aphelion.tui.state import TUIState

logger = logging.getLogger(__name__)

REFRESH_HZ = 4  # Up from 2 Hz for smoother updates


@dataclass
class TUIConfig:
    """Configuration for the Terminal User Interface."""
    refresh_rate: float = 0.25          # seconds between screen refreshes
    max_log_lines: int = 200
    show_hydra_detail: bool = True
    show_sentinel_detail: bool = True
    compact_mode: bool = False
    initial_view: str = "overview"      # overview | hydra | risk | analytics | logs
    theme: str = "bloomberg"            # bloomberg | light (reserved)
    enable_keyboard: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Textual App (preferred)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TEXTUAL:

    class _DashboardWidget(Static):
        """Auto-refreshing widget that renders the full Rich Layout."""

        def __init__(self, tui_state: TUIState, view: str = "overview", **kw):
            super().__init__(**kw)
            self._state = tui_state
            self._view = view

        @property
        def view(self) -> str:
            return self._view

        @view.setter
        def view(self, v: str) -> None:
            self._view = v
            self.refresh()

        def render(self):
            return build_dashboard_layout(self._state, view=self._view)

    class AphelionTextualApp(App):
        """
        Bloomberg-grade Textual TUI for APHELION.

        Keyboard shortcuts
        ------------------
        F1  — Overview   F2 — HYDRA detail   F3 — Risk detail
        F4  — Analytics  F5 — Full logs       Q  — Quit
        """

        TITLE = "APHELION Trading System"
        SUB_TITLE = "Bloomberg-grade Terminal"

        CSS = """
        Screen {
            background: rgb(8,8,24);
        }
        #main-dashboard {
            height: 1fr;
        }
        Footer {
            background: rgb(10,10,40);
            color: rgb(180,180,220);
        }
        Header {
            background: rgb(10,10,40);
            color: rgb(255,176,0);
            dock: top;
            height: 1;
        }
        """

        BINDINGS = [
            Binding("f1", "switch_view('overview')", "Overview", show=True),
            Binding("f2", "switch_view('hydra')", "HYDRA", show=True),
            Binding("f3", "switch_view('risk')", "Risk", show=True),
            Binding("f4", "switch_view('analytics')", "Analytics", show=True),
            Binding("f5", "switch_view('logs')", "Logs", show=True),
            Binding("f6", "switch_view('launcher')", "Launcher", show=True),
            Binding("f7", "switch_view('setup')", "Setup", show=True),
            Binding("f8", "switch_view('hephaestus')", "Forge", show=True),
            Binding("f9", "switch_view('training')", "Train", show=True),
            Binding("f10", "switch_view('backtest')", "Backtest", show=True),
            Binding("q", "quit", "Quit", show=True),
            Binding("r", "context_r", "Refresh", show=False),
            # Arrow navigation (context-sensitive)
            Binding("up", "context_up", "Up", show=False),
            Binding("down", "context_down", "Down", show=False),
            Binding("left", "context_left", "Left", show=False),
            Binding("right", "context_right", "Right", show=False),
            # Context-sensitive keys (dispatched by current view)
            Binding("enter", "context_enter", "Select", show=False),
            Binding("escape", "context_escape", "Back", show=False),
            Binding("s", "context_s", show=False),
            Binding("b", "context_b", show=False),
            Binding("c", "context_c", show=False),
            Binding("t", "context_t", show=False),
            Binding("h", "context_h", show=False),
            Binding("f", "context_f", show=False),
            Binding("d", "context_d", show=False),
            Binding("v", "context_v", show=False),
            Binding("p", "context_p", show=False),
            Binding("e", "context_e", show=False),
            Binding("w", "context_w", show=False),
            Binding("m", "context_m", show=False),
            Binding("l", "context_l", show=False),
            Binding("x", "context_x", show=False),
            Binding("tab", "context_tab", show=False),
            Binding("1", "context_1", show=False),
            Binding("2", "context_2", show=False),
            Binding("3", "context_3", show=False),
        ]

        def __init__(
            self,
            state: TUIState | None = None,
            config: TUIConfig | None = None,
            controller=None,
            **kw,
        ):
            super().__init__(**kw)
            self._state = state or TUIState()
            self._config = config or TUIConfig()
            self._dashboard: _DashboardWidget | None = None

            # Unified controller (preferred) or lazy sub-controllers
            self._controller = controller
            self._session_ctrl = None
            self._training_ctrl = None
            self._forge_ctrl = None
            self._backtest_ctrl = None

            # Live simulator for simulated mode (used when no controller)
            self._simulator = None

            # Navigation state
            self._selected_launcher_card = 0  # 0=paper, 1=simulated, 2=backtest
            self._scroll_offset = 0           # For scrollable content
            self._selected_position_idx = 0   # For positions table

        @property
        def state(self) -> TUIState:
            return self._state

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            self._dashboard = _DashboardWidget(
                self._state,
                view=self._config.initial_view,
                id="main-dashboard",
            )
            yield self._dashboard
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(self._config.refresh_rate, self._tick)

        def _tick(self) -> None:
            if self._dashboard:
                self._dashboard.refresh()

        def action_switch_view(self, view: str) -> None:
            if self._dashboard:
                self._dashboard.view = view

        def action_refresh_now(self) -> None:
            self._tick()

        # ── Helpers ──────────────────────────────────────────────────

        @property
        def _current_view(self) -> str:
            return self._dashboard._view if self._dashboard else "overview"

        def _get_config(self):
            """Return the AphelionConfig — from controller if available."""
            if self._controller is not None:
                return self._controller.config
            cfg = getattr(self._state, "_aphelion_config", None)
            if cfg is None:
                from aphelion.tui.config import AphelionConfig
                cfg = AphelionConfig()
                self._state._aphelion_config = cfg
            return cfg

        def _get_setup_form(self):
            """Return the setup FormState, creating from config if needed."""
            form = getattr(self._state, "_setup_form", None)
            if form is None:
                from aphelion.tui.screens.setup import build_setup_form
                form = build_setup_form(self._get_config())
                self._state._setup_form = form
            return form

        def _get_text_area(self):
            """Return HEPHAESTUS text area state."""
            ta = getattr(self._state, "_heph_text_area", None)
            if ta is None:
                from aphelion.tui.widgets.text_area import TextAreaState
                ta = TextAreaState()
                self._state._heph_text_area = ta
            return ta

        def _get_training_ctrl(self):
            if self._controller is not None:
                return self._controller.training_ctrl
            if self._training_ctrl is None:
                from aphelion.tui.controller import TrainingController
                self._training_ctrl = TrainingController()
            return self._training_ctrl

        def _get_forge_ctrl(self):
            if self._controller is not None:
                return self._controller.forge_ctrl
            if self._forge_ctrl is None:
                from aphelion.tui.controller import ForgeController
                self._forge_ctrl = ForgeController()
            return self._forge_ctrl

        def _get_backtest_ctrl(self):
            if self._controller is not None:
                return self._controller.backtest_ctrl
            if self._backtest_ctrl is None:
                from aphelion.tui.controller import BacktestController
                self._backtest_ctrl = BacktestController()
            return self._backtest_ctrl

        def _notify_user(self, message: str, severity: str = "information") -> None:
            """Show a Textual notification toast."""
            try:
                self.notify(message, severity=severity)
            except Exception:
                pass  # older Textual versions may not support notify

        # ── Context-sensitive key handlers ───────────────────────────

        def action_context_enter(self) -> None:
            view = self._current_view
            if view == "launcher":
                # ENTER → start session based on selected card
                card = self._selected_launcher_card
                if card == 0:
                    self._start_paper_session(mode="paper")
                elif card == 1:
                    self._start_paper_session(mode="simulated")
                else:
                    self.action_switch_view("backtest")
            elif view == "setup":
                # ENTER on active field — no-op (field editing)
                pass
            elif view == "training":
                ctrl = self._get_training_ctrl()
                from aphelion.tui.controller import TrainingState
                if ctrl.state == TrainingState.IDLE:
                    preset = getattr(self._state, "_selected_training_preset", -1)
                    from aphelion.tui.screens.training_panel import TRAINING_PRESETS
                    config = {}
                    if 0 <= preset < len(TRAINING_PRESETS):
                        p = TRAINING_PRESETS[preset]
                        config = {"epochs": p["epochs"], "data_source": p["data_source"]}
                    else:
                        config = {"epochs": 20}
                    if self._controller is not None:
                        prog = self._controller.start_training(config)
                        self._state._training_progress = prog
                    else:
                        ctrl.start_training(config)
                        self._state._training_progress = ctrl.progress
                    self._notify_user("Training started")
                elif ctrl.state == TrainingState.PAUSED:
                    if self._controller is not None:
                        self._controller.resume_training()
                    else:
                        ctrl.resume()
                    self._notify_user("Training resumed")
            elif view == "backtest":
                ctrl = self._get_backtest_ctrl()
                if not ctrl.is_active:
                    cfg = self._get_config()
                    bt_config = {
                        "symbol": cfg.backtest.symbol,
                        "start_date": cfg.backtest.start_date,
                        "end_date": cfg.backtest.end_date,
                        "capital": cfg.backtest.capital,
                    }
                    if self._controller is not None:
                        prog = self._controller.start_backtest(bt_config)
                        self._state._backtest_progress = prog
                    else:
                        ctrl.start_backtest(bt_config)
                        self._state._backtest_progress = ctrl.progress
                    self._notify_user("Backtest started")

        def action_context_escape(self) -> None:
            view = self._current_view
            if view in ("setup", "hephaestus", "training", "backtest"):
                self.action_switch_view("launcher")
            # Other views: do nothing (or could go to overview)

        def action_context_s(self) -> None:
            view = self._current_view
            if view == "launcher":
                # [S] → start simulated session
                self._start_paper_session(mode="simulated")
            elif view == "setup":
                # [S] → save config
                form = self._get_setup_form()
                cfg = self._get_config()
                from aphelion.tui.screens.setup import apply_form_to_config
                errors = apply_form_to_config(form, cfg)
                if errors:
                    self._state._setup_save_message = f"Errors: {'; '.join(errors)}"
                    self._notify_user(f"Save failed: {errors[0]}", severity="error")
                else:
                    cfg.first_run = False
                    cfg.save()
                    self._state._setup_save_message = "✓ Saved successfully"
                    self._notify_user("Configuration saved")
            elif view == "backtest":
                # [S] → save to registry (placeholder)
                self._notify_user("Results saved to registry")

        def action_context_b(self) -> None:
            view = self._current_view
            if view == "launcher":
                # [B] → switch to backtest
                self.action_switch_view("backtest")

        def action_context_c(self) -> None:
            view = self._current_view
            if view == "launcher":
                # [C] → switch to config/setup
                self.action_switch_view("setup")

        def action_context_t(self) -> None:
            view = self._current_view
            if view == "launcher":
                # [T] → switch to training
                self.action_switch_view("training")

        def action_context_h(self) -> None:
            view = self._current_view
            if view == "launcher":
                # [H] → switch to hephaestus
                self.action_switch_view("hephaestus")

        def action_context_f(self) -> None:
            view = self._current_view
            if view == "hephaestus":
                # [F] → forge strategy — try file input first, then text area
                raw_code = ""
                heph_file = os.path.join("config", "hephaestus_input.txt")
                if os.path.isfile(heph_file):
                    try:
                        with open(heph_file, "r", encoding="utf-8") as f:
                            raw_code = f.read().strip()
                    except Exception:
                        pass
                if not raw_code:
                    ta = self._get_text_area()
                    raw_code = ta.content
                if not raw_code:
                    self._notify_user("Paste code or put it in config/hephaestus_input.txt", severity="warning")
                    return

                if self._controller is not None:
                    prog = self._controller.start_forge(raw_code)
                    self._state._heph_forge_progress = prog
                else:
                    ctrl = self._get_forge_ctrl()
                    if ctrl.is_active:
                        self._notify_user("Forge already running", severity="warning")
                        return
                    ctrl.start_forge(raw_code)
                    self._state._heph_forge_progress = ctrl.progress
                self._notify_user("Forge started...")

        def action_context_d(self) -> None:
            view = self._current_view
            if view == "hephaestus":
                # [D] → deploy selected strategy (placeholder)
                self._notify_user("Strategy deployed")

        def action_context_v(self) -> None:
            view = self._current_view
            if view == "hephaestus":
                # [V] → view forge report (placeholder)
                self._notify_user("Report opened")

        def action_context_r(self) -> None:
            view = self._current_view
            if view == "setup":
                # [R] → reset config to defaults
                cfg = self._get_config()
                cfg.reset_to_defaults()
                # Rebuild form from fresh config
                from aphelion.tui.screens.setup import build_setup_form
                self._state._setup_form = build_setup_form(cfg)
                self._state._setup_save_message = "Reset to defaults"
                self._notify_user("Config reset to defaults")
            elif view == "hephaestus":
                # [R] → remove selected strategy (placeholder)
                self._notify_user("Strategy removed")
            else:
                # Default: refresh
                self._tick()

        def action_context_p(self) -> None:
            view = self._current_view
            if view == "training":
                from aphelion.tui.controller import TrainingState
                ctrl = self._get_training_ctrl()
                if ctrl.state == TrainingState.RUNNING:
                    if self._controller is not None:
                        self._controller.pause_training()
                    else:
                        ctrl.pause()
                    self._notify_user("Training paused")
            elif view == "backtest":
                self._notify_user("Report printed")

        def action_context_e(self) -> None:
            view = self._current_view
            if view == "backtest":
                # [E] → export CSV (placeholder)
                self._notify_user("Results exported to CSV")

        def action_context_w(self) -> None:
            view = self._current_view
            if view == "backtest":
                # [W] → walk-forward (placeholder)
                self._notify_user("Walk-forward analysis started")

        def action_context_m(self) -> None:
            view = self._current_view
            if view == "backtest":
                # [M] → monte carlo (placeholder)
                self._notify_user("Monte Carlo simulation started")

        def action_context_l(self) -> None:
            view = self._current_view
            if view == "training":
                # [L] → load checkpoint (placeholder)
                self._notify_user("Load checkpoint from file picker")

        def action_context_tab(self) -> None:
            view = self._current_view
            if view == "setup":
                form = self._get_setup_form()
                form.next_field()
                self._tick()

        def action_context_1(self) -> None:
            if self._current_view == "training":
                self._state._selected_training_preset = 0
                self._notify_user("Preset 1: Quick test")
                self._tick()

        def action_context_2(self) -> None:
            if self._current_view == "training":
                self._state._selected_training_preset = 1
                self._notify_user("Preset 2: Full synthetic")
                self._tick()

        def action_context_3(self) -> None:
            if self._current_view == "training":
                self._state._selected_training_preset = 2
                self._notify_user("Preset 3: Full real data")
                self._tick()

        # ── Session launch helper ────────────────────────────────────

        def _start_paper_session(self, mode: str = "paper") -> None:
            """Start a trading session and switch to the overview dashboard."""
            if self._controller is not None:
                # Route through unified controller
                self._controller.start_session(mode)
                self._notify_user(f"Session started ({mode})")
            else:
                # Legacy path — direct simulator / controller creation
                cfg = self._get_config()
                cfg.trading.mode = mode

                # Stop any existing simulator
                if self._simulator and self._simulator.is_running:
                    self._simulator.stop()
                    self._simulator = None

                if mode == "simulated":
                    from aphelion.tui.simulator import LiveSimulator
                    capital = cfg.trading.capital if hasattr(cfg.trading, "capital") else 10_000.0
                    symbol = cfg.trading.symbol if hasattr(cfg.trading, "symbol") else "XAUUSD"
                    self._simulator = LiveSimulator(self._state, cfg)
                    self._simulator.start(capital=capital, symbol=symbol)
                    self._state.session_name = f"Sim-{symbol}"
                    self._notify_user(f"Simulated session started — {symbol} — ${capital:,.0f}")
                else:
                    self._state.push_log("INFO", f"Starting {mode} session — {cfg.trading.symbol}")
                    self._state.session_name = f"Paper-{cfg.trading.symbol}"
                    try:
                        from aphelion.tui.controller import SessionController
                        if self._session_ctrl is None:
                            self._session_ctrl = SessionController()
                        from aphelion.tui.config import config_to_runner_config
                        runner_config = config_to_runner_config(cfg, mode)
                        try:
                            self.run_worker(
                                self._session_ctrl.start_session(runner_config, self._state),
                                name="paper-session",
                                exclusive=True,
                            )
                        except Exception:
                            from aphelion.tui.simulator import LiveSimulator
                            self._simulator = LiveSimulator(self._state, cfg)
                            self._simulator.start()
                    except Exception as exc:
                        logger.warning("Could not start PaperRunner, using simulator: %s", exc)
                        from aphelion.tui.simulator import LiveSimulator
                        self._simulator = LiveSimulator(self._state, cfg)
                        self._simulator.start()
                    self._notify_user(f"Starting {mode} session...")

            self.action_switch_view("overview")

        def _stop_session(self) -> None:
            """Stop the current running session/simulator."""
            if self._controller is not None:
                self._controller.stop_session()
                self._notify_user("Session stopped")
                return
            if self._simulator and self._simulator.is_running:
                self._simulator.stop()
                self._simulator = None
                self._notify_user("Session stopped")
            if self._session_ctrl:
                try:
                    import asyncio
                    asyncio.ensure_future(self._session_ctrl.stop_session())
                except Exception:
                    pass

        # ── Arrow key handlers ───────────────────────────────────────

        def action_context_up(self) -> None:
            view = self._current_view
            if view == "launcher":
                self._selected_launcher_card = max(0, self._selected_launcher_card - 1)
                self._state._launcher_selected = self._selected_launcher_card
                self._tick()
            elif view == "setup":
                form = self._get_setup_form()
                form.prev_field()
                self._tick()
            elif view == "training":
                idx = getattr(self._state, "_selected_training_preset", 0)
                self._state._selected_training_preset = max(0, idx - 1)
                self._tick()
            elif view == "overview":
                # Scroll positions
                self._selected_position_idx = max(0, self._selected_position_idx - 1)
                self._tick()
            elif view == "logs":
                self._scroll_offset = max(0, self._scroll_offset - 1)
                self._tick()
            elif view == "hephaestus":
                ta = self._get_text_area()
                ta.cursor_up()
                self._tick()
            elif view in ("hydra", "risk", "analytics", "backtest", "evolution", "sola"):
                self._scroll_offset = max(0, self._scroll_offset - 3)
                self._tick()

        def action_context_down(self) -> None:
            view = self._current_view
            if view == "launcher":
                self._selected_launcher_card = min(2, self._selected_launcher_card + 1)
                self._state._launcher_selected = self._selected_launcher_card
                self._tick()
            elif view == "setup":
                form = self._get_setup_form()
                form.next_field()
                self._tick()
            elif view == "training":
                idx = getattr(self._state, "_selected_training_preset", 0)
                self._state._selected_training_preset = min(2, idx + 1)
                self._tick()
            elif view == "overview":
                max_pos = max(0, len(self._state.positions) - 1)
                self._selected_position_idx = min(max_pos, self._selected_position_idx + 1)
                self._tick()
            elif view == "logs":
                max_scroll = max(0, len(self._state.log) - 20)
                self._scroll_offset = min(max_scroll, self._scroll_offset + 1)
                self._tick()
            elif view == "hephaestus":
                ta = self._get_text_area()
                ta.cursor_down()
                self._tick()
            elif view in ("hydra", "risk", "analytics", "backtest", "evolution", "sola"):
                self._scroll_offset += 3
                self._tick()

        def action_context_left(self) -> None:
            view = self._current_view
            if view == "overview":
                # Cycle to previous main view
                views = ["overview", "hydra", "risk", "analytics", "logs"]
                idx = views.index(view) if view in views else 0
                new_view = views[(idx - 1) % len(views)]
                self.action_switch_view(new_view)
            elif view == "setup":
                form = self._get_setup_form()
                form.prev_section()
                self._tick()
            elif view == "hephaestus":
                ta = self._get_text_area()
                ta.cursor_left()
                self._tick()
            elif view == "launcher":
                # Cycle to previous launcher card
                self._selected_launcher_card = max(0, self._selected_launcher_card - 1)
                self._state._launcher_selected = self._selected_launcher_card
                self._tick()

        def action_context_right(self) -> None:
            view = self._current_view
            if view == "overview":
                views = ["overview", "hydra", "risk", "analytics", "logs"]
                idx = views.index(view) if view in views else 0
                new_view = views[(idx + 1) % len(views)]
                self.action_switch_view(new_view)
            elif view == "setup":
                form = self._get_setup_form()
                form.next_section()
                self._tick()
            elif view == "hephaestus":
                ta = self._get_text_area()
                ta.cursor_right()
                self._tick()
            elif view == "launcher":
                self._selected_launcher_card = min(2, self._selected_launcher_card + 1)
                self._state._launcher_selected = self._selected_launcher_card
                self._tick()

        def action_context_x(self) -> None:
            """[X] — Stop session."""
            view = self._current_view
            if view in ("overview", "hydra", "risk", "analytics", "logs"):
                self._stop_session()
                self._state.push_log("INFO", "Session stopped by user")
                self._notify_user("Session stopped")
            elif view == "training":
                if self._controller:
                    self._controller.stop_training()
                else:
                    ctrl = self._get_training_ctrl()
                    ctrl.stop()
                self._notify_user("Training cancelled")
            elif view == "backtest":
                if self._controller:
                    self._controller.stop_backtest()
                else:
                    ctrl = self._get_backtest_ctrl()
                    ctrl.stop()
                self._notify_user("Backtest cancelled")


# ═══════════════════════════════════════════════════════════════════════════
# Rich-Live Fallback
# ═══════════════════════════════════════════════════════════════════════════

class _RichLiveTUI:
    """Rich Live fallback when Textual is not installed."""

    def __init__(self, state: TUIState, config: TUIConfig):
        self._state = state
        self._config = config
        self._console = Console()
        self._running = False
        self._current_view = config.initial_view

    @property
    def state(self) -> TUIState:
        return self._state

    async def run(self) -> None:
        self._running = True
        with Live(
            build_dashboard_layout(self._state, view=self._current_view),
            console=self._console,
            refresh_per_second=REFRESH_HZ,
            screen=True,
        ) as live:
            try:
                while self._running:
                    live.update(build_dashboard_layout(
                        self._state, view=self._current_view
                    ))
                    await asyncio.sleep(self._config.refresh_rate)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run_sync(self) -> None:
        asyncio.run(self.run())


# ═══════════════════════════════════════════════════════════════════════════
# Unified entry-point
# ═══════════════════════════════════════════════════════════════════════════

class AphelionTUI:
    """
    Unified APHELION TUI entry-point.

    Automatically selects Textual (interactive) or Rich Live (read-only)
    depending on installed packages.

    Usage::

        controller = AphelionController(config)
        tui = AphelionTUI(controller=controller, config=tui_config)
        tui.run_sync()
    """

    def __init__(
        self,
        state: Optional[TUIState] = None,
        config: Optional[TUIConfig] = None,
        controller=None,
    ):
        if not HAS_RICH:
            raise ImportError("rich is required for TUI. Install: pip install rich")
        self._state = state or TUIState()
        self._config = config or TUIConfig()
        self._controller = controller

        # Wire controller ↔ state
        if controller is not None:
            controller.tui_state = self._state
            # Attach config to state for screens that read it directly
            self._state._aphelion_config = controller.config

        if HAS_TEXTUAL:
            self._backend = AphelionTextualApp(
                self._state, self._config, controller=controller,
            )
        else:
            self._backend = _RichLiveTUI(self._state, self._config)

    @property
    def state(self) -> TUIState:
        return self._state

    async def run(self) -> None:
        """Run the TUI (blocking)."""
        if HAS_TEXTUAL and isinstance(self._backend, AphelionTextualApp):
            await self._backend.run_async()
        else:
            await self._backend.run()

    def stop(self) -> None:
        """Signal the TUI to stop."""
        if hasattr(self._backend, "stop"):
            self._backend.stop()
        elif hasattr(self._backend, "exit"):
            self._backend.exit()

    def run_sync(self) -> None:
        """Blocking synchronous entry-point."""
        if HAS_TEXTUAL and isinstance(self._backend, AphelionTextualApp):
            self._backend.run()
        else:
            self._backend.run_sync()
