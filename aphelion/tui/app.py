"""
APHELION TUI — Core Application
Orchestrates all panels into a Rich Live display that refreshes at ~2 Hz.
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
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from aphelion.tui.screens.dashboard import build_dashboard_layout
from aphelion.tui.state import TUIState

logger = logging.getLogger(__name__)

REFRESH_HZ = 2  # Dashboard refresh rate


@dataclass
class TUIConfig:
    """Configuration for the Terminal User Interface."""
    refresh_rate: float = 0.5  # seconds between screen refreshes
    max_log_lines: int = 100
    show_hydra_detail: bool = True
    show_sentinel_detail: bool = True
    compact_mode: bool = False


class AphelionTUI:
    """
    Rich Live terminal dashboard for APHELION.

    Displays:
      - Header with session info, clock, and market status
      - HYDRA signal panel (direction, confidence, horizon agreement)
      - SENTINEL risk panel (drawdown, circuit breakers, limits)
      - Positions table (open paper/live positions)
      - Equity / PnL summary
      - Event log (recent fills, rejections, alerts)

    Usage:
        state = TUIState()
        tui = AphelionTUI(state)
        await tui.run()  # blocks, refreshing the screen
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
        self._console = Console()
        self._running = False

    @property
    def state(self) -> TUIState:
        """Mutable state object — feed data here from the paper session loop."""
        return self._state

    async def run(self) -> None:
        """Run the live dashboard until cancelled."""
        self._running = True
        with Live(
            build_dashboard_layout(self._state),
            console=self._console,
            refresh_per_second=REFRESH_HZ,
            screen=True,
        ) as live:
            try:
                while self._running:
                    live.update(build_dashboard_layout(self._state))
                    await asyncio.sleep(self._config.refresh_rate)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
        self._running = False

    def stop(self) -> None:
        """Signal the dashboard to stop after the current frame."""
        self._running = False

    def run_sync(self) -> None:
        """Blocking synchronous entry-point (convenience wrapper)."""
        asyncio.run(self.run())
