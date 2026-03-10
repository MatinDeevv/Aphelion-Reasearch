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

# Phase 23 — deferred imports for new screens
_PHASE23_SCREENS_LOADED = False


def _lazy_import_phase23():
    """Lazy-import Phase 23 screens to avoid circular imports."""
    global _PHASE23_SCREENS_LOADED
    if _PHASE23_SCREENS_LOADED:
        return
    _PHASE23_SCREENS_LOADED = True
    global build_launcher_panel, build_setup_panel, build_hephaestus_panel
    global build_training_panel, build_backtest_panel
    global build_evolution_panel, build_sola_panel
    from aphelion.tui.screens.launcher import build_launcher_panel
    from aphelion.tui.screens.hephaestus_panel import build_hephaestus_panel
    from aphelion.tui.screens.training_panel import build_training_panel
    from aphelion.tui.screens.backtest_panel import build_backtest_panel
    from aphelion.tui.screens.evolution_panel import build_evolution_panel
    from aphelion.tui.screens.sola_panel import build_sola_panel


def _build_session_control_bar(state: TUIState) -> Panel:
    """Session control bar shown at the top of overview when a session is active."""
    bar = Text()
    bar.append("  ● SESSION ACTIVE", style="bold bright_green")
    bar.append(f"  {state.session_name}", style="bright_white")
    bar.append(f"  │  Mode: {state.feed_mode}", style="dim")
    if state.bars_processed:
        bar.append(f"  │  Bars: {state.bars_processed:,}", style="dim")
    if state.equity.total_trades:
        wr = state.win_rate * 100
        bar.append(f"  │  Trades: {state.equity.total_trades}", style="dim")
        bar.append(f"  WR: {wr:.0f}%", style="bright_green" if wr >= 50 else "bright_red")
    pnl = state.equity.daily_pnl
    pnl_style = "bright_green" if pnl >= 0 else "bright_red"
    bar.append(f"  │  PnL: ${pnl:+,.0f}", style=pnl_style)
    bar.append("  │  ", style="dim")
    bar.append("[X]", style="bold bright_cyan")
    bar.append(" Stop", style="dim")
    return Panel(bar, border_style="bright_green", height=3)


def _build_no_session_bar() -> Panel:
    """Placeholder bar shown on overview when no session is running."""
    bar = Text()
    bar.append("  ○ NO SESSION", style="dim")
    bar.append("  │  Press ", style="dim")
    bar.append("[F6]", style="bold bright_cyan")
    bar.append(" to launch", style="dim")
    return Panel(bar, border_style="dim", height=3)


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
    bar.append(" F6", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Launch ", style="dim")
    bar.append(" F7", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Setup ", style="dim")
    bar.append(" F8", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Forge ", style="dim")
    bar.append(" F9", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Train ", style="dim")
    bar.append(" F10", style="bold bright_white on rgb(0,80,180)")
    bar.append(":Backtest ", style="dim")
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

    elif view == "launcher":
        _lazy_import_phase23()
        config = getattr(state, "_aphelion_config", None)
        if config is None:
            from aphelion.tui.config import AphelionConfig
            config = AphelionConfig()
        last_session = getattr(state, "_last_session_summary", None)
        selected_card = getattr(state, "_launcher_selected", 0)
        layout.split_column(
            Layout(name="launcher", ratio=1),
        )
        layout["launcher"].update(build_launcher_panel(config, last_session, selected_card))

    elif view == "setup":
        _lazy_import_phase23()
        from aphelion.tui.screens.setup import build_setup_panel
        form = getattr(state, "_setup_form", None)
        config = getattr(state, "_aphelion_config", None)
        if config is None:
            from aphelion.tui.config import AphelionConfig
            config = AphelionConfig()
        if form is None:
            from aphelion.tui.screens.setup import build_setup_form
            form = build_setup_form(config)
            state._setup_form = form
        msg = getattr(state, "_setup_save_message", "")
        layout.split_column(
            Layout(name="setup", ratio=1),
        )
        layout["setup"].update(build_setup_panel(form, config, msg))

    elif view == "hephaestus":
        _lazy_import_phase23()
        ta = getattr(state, "_heph_text_area", None)
        if ta is None:
            from aphelion.tui.widgets.text_area import TextAreaState
            ta = TextAreaState()
            state._heph_text_area = ta
        fp = getattr(state, "_heph_forge_progress", None)
        deployed = getattr(state, "_heph_deployed", [])
        rejections = getattr(state, "_heph_rejections", [])
        layout.split_column(
            Layout(name="hephaestus", ratio=1),
        )
        layout["hephaestus"].update(build_hephaestus_panel(ta, fp, deployed, rejections))

    elif view == "training":
        _lazy_import_phase23()
        tp = getattr(state, "_training_progress", None)
        layout.split_column(
            Layout(name="training", ratio=1),
        )
        layout["training"].update(build_training_panel(tp))

    elif view == "backtest":
        _lazy_import_phase23()
        bp = getattr(state, "_backtest_progress", None)
        bt_config = getattr(state, "_backtest_config", None)
        eq_curve = getattr(state, "_backtest_equity_curve", None)
        layout.split_column(
            Layout(name="backtest", ratio=1),
        )
        layout["backtest"].update(build_backtest_panel(bp, bt_config, eq_curve))

    elif view == "evolution":
        _lazy_import_phase23()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="evolution", ratio=4),
            Layout(name="status", size=1),
        )
        layout["header"].update(build_header(state))
        layout["evolution"].update(build_evolution_panel(state))
        layout["status"].update(_build_status_bar(state))

    elif view == "sola":
        _lazy_import_phase23()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="sola", ratio=4),
            Layout(name="status", size=1),
        )
        layout["header"].update(build_header(state))
        layout["sola"].update(build_sola_panel(state))
        layout["status"].update(_build_status_bar(state))

    else:
        # Default overview layout — with session control bar
        has_session = state.market_open or state.feed_connected

        if has_session:
            layout.split_column(
                Layout(name="header", size=4),
                Layout(name="session_bar", size=3),
                Layout(name="middle", ratio=3),
                Layout(name="positions", ratio=2),
                Layout(name="bottom", ratio=2),
                Layout(name="status", size=1),
            )
            layout["session_bar"].update(_build_session_control_bar(state))
        else:
            layout.split_column(
                Layout(name="header", size=4),
                Layout(name="session_bar", size=3),
                Layout(name="middle", ratio=3),
                Layout(name="positions", ratio=2),
                Layout(name="bottom", ratio=2),
                Layout(name="status", size=1),
            )
            layout["session_bar"].update(_build_no_session_bar())

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
