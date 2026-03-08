"""
APHELION TUI — Equity / PnL Panel
Shows account equity, daily PnL, win rate, trade count.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState


def build_equity_panel(state: TUIState) -> Panel:
    """Build the equity / PnL summary panel."""
    e = state.equity

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column()

    # Equity
    eq_line = Text(f"  Equity: ${e.account_equity:,.2f}", style="bold white")
    table.add_row(eq_line)

    # Daily PnL
    pnl_style = "green" if e.daily_pnl >= 0 else "red"
    pnl_line = Text(f"  Daily PnL: ${e.daily_pnl:+,.2f}", style=pnl_style)
    table.add_row(pnl_line)

    # Realized / Unrealized
    r_style = "green" if e.realized_pnl >= 0 else "red"
    u_style = "green" if e.unrealized_pnl >= 0 else "red"
    realized = Text(f"  Realized: ${e.realized_pnl:+,.2f}", style=r_style)
    unrealized = Text(f"  Unrealized: ${e.unrealized_pnl:+,.2f}", style=u_style)
    table.add_row(realized)
    table.add_row(unrealized)

    # Trade stats
    total = e.total_trades
    win_rate = (e.winning_trades / total * 100) if total > 0 else 0.0
    stats_line = Text(
        f"  Trades: {total}  W: {e.winning_trades}  L: {e.losing_trades}  "
        f"WR: {win_rate:.1f}%",
        style="white",
    )
    table.add_row(stats_line)

    return Panel(
        table,
        title="[bold green]💰 Equity[/]",
        border_style="green",
    )
