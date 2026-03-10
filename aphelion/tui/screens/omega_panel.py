"""
APHELION TUI — OMEGA Strategy Panel  (v2 — Bloomberg-grade)

Displays OMEGA H1/H4 swing strategy status: trend state,
open swing positions, allocation, and performance.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.gauge import render_gauge


def build_omega_panel(state: TUIState) -> Panel:
    """Build the OMEGA strategy panel."""

    # OMEGA data from extended state
    trend = getattr(state, "omega_trend", "FLAT")
    trend_strength = getattr(state, "omega_trend_strength", 0.0)
    allocation_pct = getattr(state, "omega_allocation_pct", 0.30)
    omega_wr = getattr(state, "omega_win_rate", 0.0)
    omega_sharpe = getattr(state, "omega_sharpe", 0.0)
    omega_pnl = getattr(state, "omega_pnl_today", 0.0)
    omega_trades = getattr(state, "omega_trades_today", 0)
    omega_positions = getattr(state, "omega_open_positions", [])

    # ── Trend banner ──
    trend_styles = {
        "BULL": ("bright_green", "▲"),
        "BEAR": ("bright_red", "▼"),
        "FLAT": ("bright_yellow", "─"),
    }
    style, arrow = trend_styles.get(trend, ("dim", "?"))

    banner = Text()
    banner.append(f"  {arrow} TREND: {trend}  ", style=f"bold {style}")
    banner.append(f"  strength={trend_strength:.0%}", style="dim")

    # ── Gauges ──
    gauges = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    gauges.add_column("label", ratio=1)
    gauges.add_column("gauge", ratio=3)

    gauges.add_row(
        Text("  Alloc", style="bright_white"),
        render_gauge(allocation_pct * 100, 100.0, width=22, label=f"{allocation_pct:.0%}"),
    )

    # ── Stats ──
    stats = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    stats.add_column("k", ratio=1)
    stats.add_column("v", ratio=2)

    pnl_style = "bright_green" if omega_pnl >= 0 else "bright_red"
    stats.add_row(Text("  Win Rate", style="bright_white"), Text(f"{omega_wr:.1%}", style="bright_cyan"))
    stats.add_row(Text("  Sharpe", style="bright_white"), Text(f"{omega_sharpe:.2f}", style="bright_cyan"))
    stats.add_row(Text("  Today PnL", style="bright_white"), Text(f"${omega_pnl:+.2f}", style=pnl_style))
    stats.add_row(Text("  Trades", style="bright_white"), Text(str(omega_trades), style="bright_cyan"))

    # ── Open swing positions ──
    pos_table = Table(show_header=True, expand=True, show_edge=False, padding=(0, 1))
    pos_table.add_column("Dir", width=5)
    pos_table.add_column("Entry", justify="right")
    pos_table.add_column("SL", justify="right")
    pos_table.add_column("TP", justify="right")
    pos_table.add_column("PnL", justify="right")

    for p in omega_positions[:5]:
        d = getattr(p, "direction", "?")
        entry = getattr(p, "entry_price", 0)
        sl = getattr(p, "stop_loss", 0)
        tp = getattr(p, "take_profit", 0)
        pnl = getattr(p, "unrealized_pnl", 0)
        pstyle = "bright_green" if pnl >= 0 else "bright_red"
        pos_table.add_row(d, f"{entry:.2f}", f"{sl:.2f}", f"{tp:.2f}", Text(f"${pnl:+.2f}", style=pstyle))

    if not omega_positions:
        pos_table.add_row("—", "—", "—", "—", Text("No positions", style="dim"))

    content = Group(banner, Text(""), gauges, Text(""), stats, Text(""), pos_table)
    return Panel(
        content,
        title="[bold bright_white]OMEGA H1/H4 SWING[/]",
        border_style="bright_yellow",
        expand=True,
    )
