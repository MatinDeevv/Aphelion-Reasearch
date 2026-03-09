"""APHELION TUI - Equity / PnL Panel  (v2 - Bloomberg-grade)

Equity dashboard with sparkline curve, PnL breakdown, performance metrics,
and color-coded trade statistics.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.sparkline import render_sparkline
from aphelion.tui.widgets.gauge import render_gauge


def build_equity_panel(state: TUIState) -> Panel:
    """Build the equity / PnL summary panel (Bloomberg-grade)."""
    e = state.equity

    headline = Text()
    headline.append("  Equity: ", style="bright_white")
    headline.append(f"${e.account_equity:,.2f}", style="bold bright_white")
    headline.append("  Peak: ", style="dim")
    headline.append(f"${e.session_peak:,.2f}", style="dim bright_cyan")

    spark_line = Text("  ")
    hist = list(e.equity_history)
    if hist:
        spark_line.append_text(render_sparkline(hist, width=50))
    else:
        spark_line.append("\u2500" * 50, style="dim")

    pnl_table = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    pnl_table.add_column(ratio=1)
    pnl_table.add_column(ratio=1)
    pnl_table.add_column(ratio=1)

    pnl_style = "bold bright_green" if e.daily_pnl >= 0 else "bold bright_red"
    daily = Text()
    daily.append("Daily: ", style="bright_white")
    daily.append(f"${e.daily_pnl:+,.2f}", style=pnl_style)

    r_style = "bright_green" if e.realized_pnl >= 0 else "bright_red"
    realized = Text()
    realized.append("Real: ", style="bright_white")
    realized.append(f"${e.realized_pnl:+,.2f}", style=r_style)

    u_style = "bright_green" if e.unrealized_pnl >= 0 else "bright_red"
    unrealized = Text()
    unrealized.append("Unreal: ", style="bright_white")
    unrealized.append(f"${e.unrealized_pnl:+,.2f}", style=u_style)

    pnl_table.add_row(daily, realized, unrealized)

    stats_table = Table(show_header=True, header_style="bold bright_cyan",
                        expand=True, show_edge=False, padding=(0, 1))
    stats_table.add_column("Trades", justify="center")
    stats_table.add_column("Wins", justify="center")
    stats_table.add_column("Losses", justify="center")
    stats_table.add_column("Win Rate", justify="center")
    stats_table.add_column("Max DD", justify="center")
    stats_table.add_column("PF", justify="center")
    stats_table.add_column("Sharpe", justify="center")

    total = e.total_trades
    win_rate = (e.winning_trades / total * 100) if total > 0 else 0.0
    wr_style = "bright_green" if win_rate >= 50 else "bright_red"
    pf_style = "bright_green" if e.profit_factor >= 1.5 else (
        "bright_yellow" if e.profit_factor >= 1.0 else "bright_red"
    )
    sharpe_style = "bright_green" if e.sharpe_ratio >= 2.0 else (
        "bright_yellow" if e.sharpe_ratio >= 1.0 else "bright_red"
    )

    stats_table.add_row(
        str(total),
        Text(str(e.winning_trades), style="bright_green"),
        Text(str(e.losing_trades), style="bright_red"),
        Text(f"{win_rate:.1f}%", style=wr_style),
        Text(f"{e.max_drawdown_pct * 100:.2f}%", style="bright_red"),
        Text(f"{e.profit_factor:.2f}", style=pf_style),
        Text(f"{e.sharpe_ratio:.2f}", style=sharpe_style),
    )

    streaks = Text("  ")
    if e.consecutive_wins > 0:
        streaks.append(f"Win streak: {e.consecutive_wins} ", style="bright_green")
    if e.consecutive_losses > 0:
        streaks.append(f"Loss streak: {e.consecutive_losses} ", style="bright_red")
    if e.best_trade != 0:
        streaks.append(f" Best: ${e.best_trade:+,.2f}", style="bright_green")
    if e.worst_trade != 0:
        streaks.append(f" Worst: ${e.worst_trade:+,.2f}", style="bright_red")

    content = Group(headline, spark_line, pnl_table, stats_table, streaks)

    return Panel(
        content,
        title="[bold rgb(255,176,0)]Equity & Performance[/]",
        border_style="rgb(255,176,0)",
    )
