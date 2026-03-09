"""APHELION TUI - Positions Table  (v2 - Bloomberg-grade)

Dense data grid with color-coded PnL, R:R ratio, hold time,
and total exposure summary.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.gauge import render_mini_bar


def build_positions_panel(state: TUIState) -> Panel:
    """Build a table of open positions (Bloomberg-grade)."""
    table = Table(
        expand=True,
        show_lines=False,
        padding=(0, 1),
        header_style="bold bright_cyan",
        show_edge=False,
    )
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Sym", max_width=8)
    table.add_column("Dir", justify="center", max_width=5)
    table.add_column("Entry", justify="right")
    table.add_column("Now", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("Lots", justify="right", max_width=6)
    table.add_column("PnL", justify="right")
    table.add_column("R:R", justify="center", max_width=5)
    table.add_column("Bars", justify="right", max_width=5)

    if not state.positions:
        table.add_row(
            "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014",
            Text("No open positions", style="dim"), "\u2014", "\u2014",
        )
    else:
        for p in state.positions:
            dir_icon = "\u25b2" if p.direction == "LONG" else "\u25bc"
            dir_style = "bold bright_green" if p.direction == "LONG" else "bold bright_red"
            pnl_style = "bold bright_green" if p.unrealized_pnl >= 0 else "bold bright_red"
            pnl_str = f"${p.unrealized_pnl:+,.2f}"

            rr_str = f"{p.risk_reward:.1f}" if p.risk_reward > 0 else "\u2014"
            rr_style = "bright_green" if p.risk_reward >= 2.0 else (
                "bright_yellow" if p.risk_reward >= 1.0 else "bright_red"
            )

            table.add_row(
                p.position_id[:8],
                Text(p.symbol, style="bright_white"),
                Text(f"{dir_icon}{p.direction}", style=dir_style),
                f"{p.entry_price:,.2f}",
                f"{p.current_price:,.2f}",
                f"{p.stop_loss:,.2f}",
                f"{p.take_profit:,.2f}",
                f"{p.size_lots:.2f}",
                Text(pnl_str, style=pnl_style),
                Text(rr_str, style=rr_style),
                str(p.hold_bars) if p.hold_bars > 0 else "\u2014",
            )

    total_pnl = sum(p.unrealized_pnl for p in state.positions)
    total_lots = sum(p.size_lots for p in state.positions)
    summary = Text()
    summary.append(f"  Total: {len(state.positions)} pos", style="bright_white")
    summary.append(f"  Lots: {total_lots:.2f}", style="bright_white")
    pnl_s = "bright_green" if total_pnl >= 0 else "bright_red"
    summary.append(f"  PnL: ${total_pnl:+,.2f}", style=pnl_s)

    content = Group(table, summary)

    return Panel(
        content,
        title="[bold bright_cyan]Open Positions[/]",
        border_style="bright_cyan",
    )
