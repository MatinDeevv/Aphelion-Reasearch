"""
APHELION TUI — Dashboard Compositor  (v2 — Bloomberg-grade)

Builds the full Rich Layout by combining all screen panels.
Supports multiple view modes: OVERVIEW, HYDRA_DETAIL, RISK_DETAIL, ANALYTICS.

Layout (vertical slices — Overview mode):
  ┌───────────────────────────────────────────────────────┐
  │  HEADER  (price ticker, session, clock, system stats) │
  ├────────────────────────┬──────────────────────────────┤
  │  HYDRA Signal Panel    │  SENTINEL Risk Panel         │
  ├────────────────────────┴──────────────────────────────┤
  │  POSITIONS TABLE  (with R:R, hold time, exposure)     │
  ├────────────────────────┬──────────────────────────────┤
  │  EQUITY / SPARKLINE    │  EVENT LOG                   │
  └────────────────────────┴──────────────────────────────┘
"""

from __future__ import annotations

from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.screens.header import build_header
from aphelion.tui.screens.hydra_panel import build_hydra_panel
from aphelion.tui.screens.sentinel_panel import build_sentinel_panel
from aphelion.tui.screens.positions import build_positions_panel
from aphelion.tui.screens.equity import build_equity_panel
from aphelion.tui.screens.event_log import build_log_panel
from aphelion.tui.screens.performance import build_performance_panel


def _build_status_bar(state: TUIState) -> Text:
    """Build the bottom status bar with keyboard shortcuts."""
    bar = Text()
    bar.append(" Q", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Quit ", style="dim")
    bar.append(" F1", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Overview ", style="dim")
    bar.append(" F2", style="bold bright_white on rgb(0,80,180)")
    bar.append(":HYDRA ", style="dim")
    bar.append(" F3", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Risk ", style="dim")
    bar.append(" F4", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Analytics ", style="dim")
    bar.append(" F5", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Logs ", style="dim")
    bar.append("  │  ", style="dim")
    wr = state.win_rate * 100
    bar.append(f"WR:{wr:.0f}%", style="bright_green" if wr >= 50 else "bright_red")
    bar.append("  ", style="dim")
    pnl = state.equity.daily_pnl
    pnl_s = "bright_green" if pnl >= 0 else "bright_red"
    bar.append(f"PnL:${pnl:+,.0f}", style=pnl_s)
    return bar


def build_dashboard_layout(state: TUIState, view: str = "overview") -> Layout:
    """
    Compose the full dashboard layout from sub-panels.

    Parameters
    ----------
    state : TUIState
    view : str
        One of: overview, hydra, risk, analytics, logs
    """
    layout = Layout()

    if view == "hydra":
        # Full-screen HYDRA detail
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="hydra", ratio=4),
            Layout(name="status", size=1),
        )
        layout["header"].update(build_header(state))
        layout["hydra"].update(build_hydra_panel(state))
        layout["status"].update(_build_status_bar(state))

    elif view == "risk":
        # Full-screen Risk detail
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="risk", ratio=2),
            Layout(name="positions", ratio=2),
            Layout(name="status", size=1),
        )
        layout["header"].update(build_header(state))
        layout["risk"].update(build_sentinel_panel(state))
        layout["positions"].update(build_positions_panel(state))
        layout["status"].update(_build_status_bar(state))

    elif view == "analytics":
        # Performance analytics
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="perf", ratio=3),
            Layout(name="equity", ratio=2),
            Layout(name="status", size=1),
        )
        layout["header"].update(build_header(state))
        layout["perf"].update(build_performance_panel(state))
        layout["equity"].update(build_equity_panel(state))
        layout["status"].update(_build_status_bar(state))

    elif view == "logs":
        # Full-screen logs
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="logs", ratio=4),
            Layout(name="status", size=1),
        )
        layout["header"].update(build_header(state))
        layout["logs"].update(build_log_panel(state, max_visible=40))
        layout["status"].update(_build_status_bar(state))

    else:
        # Default overview layout
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="middle", ratio=3),
            Layout(name="positions", ratio=2),
            Layout(name="bottom", ratio=2),
            Layout(name="status", size=1),
        )

        layout["header"].update(build_header(state))

        layout["middle"].split_row(
            Layout(name="hydra", ratio=1),
            Layout(name="sentinel", ratio=1),
        )
        layout["hydra"].update(build_hydra_panel(state))
        layout["sentinel"].update(build_sentinel_panel(state))

        layout["positions"].update(build_positions_panel(state))

        layout["bottom"].split_row(
            Layout(name="equity", ratio=1),
            Layout(name="logs", ratio=2),
        )
        layout["equity"].update(build_equity_panel(state))
        layout["logs"].update(build_log_panel(state))

        layout["status"].update(_build_status_bar(state))

    return layout
