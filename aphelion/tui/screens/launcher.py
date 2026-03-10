"""
APHELION TUI — Launcher Screen (F6)  [Phase 23]

The first screen a user sees.  Shows system status, last session summary,
and mode selection buttons.  No session is running yet — the user picks
a mode and presses a key to start.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.config import AphelionConfig


# ─── ASCII banner ────────────────────────────────────────────────────────────

BANNER = r"""
     █████╗ ██████╗ ██╗  ██╗███████╗██╗     ██╗ ██████╗ ███╗   ██╗
    ██╔══██╗██╔══██╗██║  ██║██╔════╝██║     ██║██╔═══██╗████╗  ██║
    ███████║██████╔╝███████║█████╗  ██║     ██║██║   ██║██╔██╗ ██║
    ██╔══██║██╔═══╝ ██╔══██║██╔══╝  ██║     ██║██║   ██║██║╚██╗██║
    ██║  ██║██║     ██║  ██║███████╗███████╗██║╚██████╔╝██║ ╚████║
    ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                AUTONOMOUS  XAU/USD  TRADING  SYSTEM
"""


def _check_hydra_status(config: AphelionConfig) -> tuple[str, str]:
    """Check if HYDRA checkpoint exists."""
    if config.has_hydra_checkpoint():
        return "✅ Checkpoint loaded", "bright_green"
    return "❌ No checkpoint", "bright_red"


def _check_mt5_status(config: AphelionConfig) -> tuple[str, str]:
    """Check MT5 connection readiness."""
    if config.has_mt5_credentials():
        return f"✅ Configured ({config.mt5.server})", "bright_green"
    return "⚠ Not configured", "bright_yellow"


def _check_sentinel_status() -> tuple[str, str]:
    return "✅ Online", "bright_green"


def _check_sola_status() -> tuple[str, str]:
    return "✅ Active", "bright_green"


def build_launcher_panel(
    config: AphelionConfig,
    last_session: dict | None = None,
    selected_card: int = 0,
) -> Panel:
    """Build the F6 Launcher screen.

    Parameters
    ----------
    config : AphelionConfig
        Current user configuration.
    last_session : dict, optional
        Summary of the last trading session (date, capital, pnl, trades, wr).
    selected_card : int
        Index of selected mode card (0=paper, 1=simulated, 2=backtest).
    """
    parts: list = []

    # ── Banner ──
    banner = Text(BANNER, style="bold rgb(255,176,0)")
    parts.append(Align.center(banner))
    parts.append(Text(""))

    # ── System Status + Last Session ──
    status_table = Table(show_header=False, expand=True, show_edge=False, padding=(0, 2))
    status_table.add_column("section", ratio=1)
    status_table.add_column("details", ratio=1)

    # Left: System Status
    sys_text = Text()
    sys_text.append("  SYSTEM STATUS\n", style="bold bright_yellow")
    sys_text.append("  " + "─" * 35 + "\n", style="dim")

    hydra_msg, hydra_style = _check_hydra_status(config)
    mt5_msg, mt5_style = _check_mt5_status(config)
    sentinel_msg, sentinel_style = _check_sentinel_status()
    sola_msg, sola_style = _check_sola_status()

    sys_text.append("  HYDRA:    ", style="bright_white")
    sys_text.append(f"{hydra_msg}\n", style=hydra_style)
    sys_text.append("  MT5:      ", style="bright_white")
    sys_text.append(f"{mt5_msg}\n", style=mt5_style)
    sys_text.append("  SENTINEL: ", style="bright_white")
    sys_text.append(f"{sentinel_msg}\n", style=sentinel_style)
    sys_text.append("  SOLA:     ", style="bright_white")
    sys_text.append(f"{sola_msg}\n", style=sola_style)

    # Right: Last Session
    sess_text = Text()
    sess_text.append("  LAST SESSION\n", style="bold bright_yellow")
    sess_text.append("  " + "─" * 35 + "\n", style="dim")

    if last_session:
        sess_text.append(f"  Date:    {last_session.get('date', 'N/A')}\n", style="bright_white")
        capital = last_session.get("capital", 0)
        sess_text.append(f"  Capital: ${capital:,.2f}\n", style="bright_white")
        pnl = last_session.get("pnl", 0)
        pnl_pct = last_session.get("pnl_pct", 0)
        pnl_style = "bright_green" if pnl >= 0 else "bright_red"
        sess_text.append(f"  PnL:     ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n", style=pnl_style)
        trades = last_session.get("trades", 0)
        wr = last_session.get("wr", 0)
        sess_text.append(f"  Trades:  {trades} ({wr:.0f}% WR)\n", style="bright_white")
    else:
        sess_text.append("  No previous session\n", style="dim")

    status_table.add_row(sys_text, sess_text)
    parts.append(status_table)
    parts.append(Text(""))

    # ── Mode Selection Cards ──
    mode_header = Text("  SELECT MODE  ", style="bold bright_yellow")
    mode_header.append("(↑↓ or ←→ to select, ENTER to start)", style="dim")
    parts.append(mode_header)
    parts.append(Text(""))

    # Card styles based on selection
    live_border = "bold bright_green" if selected_card == 0 else "dim green"
    sim_border = "bold bright_cyan" if selected_card == 1 else "dim cyan"
    bt_border = "bold bright_magenta" if selected_card == 2 else "dim magenta"

    live_marker = "▶ " if selected_card == 0 else "  "
    sim_marker = "▶ " if selected_card == 1 else "  "
    bt_marker = "▶ " if selected_card == 2 else "  "

    card_live = Panel(
        Text.from_markup(
            f"[bold bright_white]{live_marker}[ENTER] LIVE PAPER[/]\n\n"
            "[bright_white]Real MT5 data.\n"
            f"{config.mt5.server or 'Not configured'}.\n"
            "0.01 lot min.[/]"
        ),
        border_style=live_border,
        width=26,
    )
    card_sim = Panel(
        Text.from_markup(
            f"[bold bright_white]{sim_marker}[S] SIMULATED[/]\n\n"
            "[bright_white]Fake price feed.\n"
            "No MT5 needed.\n"
            "Instant bars.[/]"
        ),
        border_style=sim_border,
        width=26,
    )
    card_bt = Panel(
        Text.from_markup(
            f"[bold bright_white]{bt_marker}[B] BACKTEST[/]\n\n"
            "[bright_white]Historical run.\n"
            "No live feed.\n"
            "Full metrics.[/]"
        ),
        border_style=bt_border,
        width=26,
    )
    parts.append(Columns([card_live, card_sim, card_bt], padding=2))
    parts.append(Text(""))

    # ── Bottom shortcuts ──
    shortcuts = Text()
    shortcuts.append("  [C]", style="bold bright_cyan")
    shortcuts.append(" Configure  ", style="dim")
    shortcuts.append("[T]", style="bold bright_cyan")
    shortcuts.append(" Train HYDRA  ", style="dim")
    shortcuts.append("[H]", style="bold bright_cyan")
    shortcuts.append(" HEPHAESTUS Forge  ", style="dim")
    shortcuts.append("[Q]", style="bold bright_cyan")
    shortcuts.append(" Quit", style="dim")
    parts.append(shortcuts)

    content = Group(*parts)

    return Panel(
        content,
        title="[bold rgb(255,176,0)]APHELION[/]",
        subtitle="[dim]Phase 23 — TUI Supreme[/]",
        border_style="rgb(255,176,0)",
        expand=True,
    )
