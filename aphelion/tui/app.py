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
            Binding("q", "quit", "Quit", show=True),
            Binding("r", "refresh_now", "Refresh", show=False),
        ]

        def __init__(
            self,
            state: TUIState | None = None,
            config: TUIConfig | None = None,
            **kw,
        ):
            super().__init__(**kw)
            self._state = state or TUIState()
            self._config = config or TUIConfig()
            self._dashboard: _DashboardWidget | None = None

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

        state = TUIState()
        tui = AphelionTUI(state)
        await tui.run()   # or  tui.run_sync()
    """

    def __init__(
        self,
        state: Optional[TUIState] = None,
        config: Optional[TUIConfig] = None,
    ):
        if not HAS_RICH:
            raise ImportError("rich is required for TUI. Install: pip install rich")
        self._state = state or TUIState()
        self._config = config or TUIConfig()

        if HAS_TEXTUAL:
            self._backend = AphelionTextualApp(self._state, self._config)
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
        asyncio.run(self.run())
