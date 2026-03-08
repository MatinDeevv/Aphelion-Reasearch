"""
APHELION TUI — Positions Table
Shows all open paper/live positions with PnL colouring.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState


def build_positions_panel(state: TUIState) -> Panel:
    """Build a table of open positions."""
    table = Table(expand=True, show_lines=False, padding=(0, 1))
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Dir", justify="center", max_width=6)
    table.add_column("Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("PnL", justify="right")

    if not state.positions:
        table.add_row("—", "—", "—", "—", "—", "—", "—", "No open positions")
    else:
        for p in state.positions:
            dir_style = "green" if p.direction == "LONG" else "red"
            pnl_style = "green" if p.unrealized_pnl >= 0 else "red"
            pnl_str = f"${p.unrealized_pnl:+,.2f}"

            table.add_row(
                p.position_id[:12],
                Text(p.direction, style=dir_style),
                f"{p.entry_price:,.2f}",
                f"{p.current_price:,.2f}",
                f"{p.stop_loss:,.2f}",
                f"{p.take_profit:,.2f}",
                f"{p.size_lots:.2f}",
                Text(pnl_str, style=pnl_style),
            )

    return Panel(
        table,
        title="[bold white]📊 Open Positions[/]",
        border_style="bright_blue",
    )
