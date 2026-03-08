"""
APHELION TUI — Dashboard Compositor
Builds the full Rich Layout by combining all screen panels.

Layout (vertical slices):
  ┌─────────────────────────────────────────────────┐
  │  HEADER  (session info, clock, market status)   │
  ├───────────────────────┬─────────────────────────┤
  │  HYDRA Signal Panel   │  SENTINEL Risk Panel    │
  ├───────────────────────┴─────────────────────────┤
  │  POSITIONS TABLE                                │
  ├───────────────────────┬─────────────────────────┤
  │  EQUITY / PnL         │  EVENT LOG              │
  └───────────────────────┴─────────────────────────┘
"""

from __future__ import annotations

from rich.layout import Layout
from rich.panel import Panel

from aphelion.tui.state import TUIState
from aphelion.tui.screens.header import build_header
from aphelion.tui.screens.hydra_panel import build_hydra_panel
from aphelion.tui.screens.sentinel_panel import build_sentinel_panel
from aphelion.tui.screens.positions import build_positions_panel
from aphelion.tui.screens.equity import build_equity_panel
from aphelion.tui.screens.event_log import build_log_panel


def build_dashboard_layout(state: TUIState) -> Layout:
    """Compose the full dashboard layout from sub-panels."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", ratio=3),
        Layout(name="positions", ratio=2),
        Layout(name="bottom", ratio=2),
    )

    # Header
    layout["header"].update(build_header(state))

    # Middle row: HYDRA + SENTINEL side by side
    layout["middle"].split_row(
        Layout(name="hydra", ratio=1),
        Layout(name="sentinel", ratio=1),
    )
    layout["hydra"].update(build_hydra_panel(state))
    layout["sentinel"].update(build_sentinel_panel(state))

    # Positions table
    layout["positions"].update(build_positions_panel(state))

    # Bottom row: Equity + Logs
    layout["bottom"].split_row(
        Layout(name="equity", ratio=1),
        Layout(name="logs", ratio=2),
    )
    layout["equity"].update(build_equity_panel(state))
    layout["logs"].update(build_log_panel(state))

    return layout
