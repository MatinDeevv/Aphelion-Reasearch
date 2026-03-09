"""APHELION TUI - Performance Analytics Screen  (v2 - Bloomberg-grade)

Detailed performance analytics: trade distribution, risk metrics,
equity curve analysis, session statistics.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.sparkline import render_sparkline
from aphelion.tui.widgets.gauge import render_gauge


def build_performance_panel(state: TUIState) -> Panel:
    """Build the performance analytics panel."""
    e = state.equity

    # -- Key metrics grid --
    metrics = Table(
        show_header=True,
        header_style="bold bright_cyan",
        expand=True,
        show_edge=False,
        padding=(0, 1),
    )
    metrics.add_column("Metric", style="bright_white", ratio=2)
    metrics.add_column("Value", justify="right", ratio=1)
    metrics.add_column("Status", justify="center", ratio=1)

    total = e.total_trades
    win_rate = (e.winning_trades / total * 100) if total > 0 else 0.0

    def _status(val: float, good: float, warn: float, higher_is_better: bool = True) -> Text:
        if higher_is_better:
            if val >= good:
                return Text("\u25cf GOOD", style="bold bright_green")
            elif val >= warn:
                return Text("\u25cf OK", style="bold bright_yellow")
            else:
                return Text("\u25cf POOR", style="bold bright_red")
        else:
            if val <= good:
                return Text("\u25cf GOOD", style="bold bright_green")
            elif val <= warn:
                return Text("\u25cf OK", style="bold bright_yellow")
            else:
                return Text("\u25cf POOR", style="bold bright_red")

    metrics.add_row("Total Trades", str(total), Text("\u00b7", style="dim"))
    metrics.add_row("Win Rate", f"{win_rate:.1f}%", _status(win_rate, 55.0, 45.0))
    metrics.add_row("Profit Factor", f"{e.profit_factor:.2f}", _status(e.profit_factor, 1.5, 1.0))
    metrics.add_row("Sharpe Ratio", f"{e.sharpe_ratio:.2f}", _status(e.sharpe_ratio, 2.0, 1.0))
    metrics.add_row("Max Drawdown", f"{e.max_drawdown_pct * 100:.2f}%",
                     _status(e.max_drawdown_pct * 100, 3.0, 5.0, higher_is_better=False))
    metrics.add_row("Avg Win", f"${e.avg_win:+,.2f}", Text("\u00b7", style="dim"))
    metrics.add_row("Avg Loss", f"${e.avg_loss:+,.2f}", Text("\u00b7", style="dim"))
    metrics.add_row("Best Trade", f"${e.best_trade:+,.2f}",
                     Text("\u25b2", style="bright_green") if e.best_trade > 0 else Text("\u00b7", style="dim"))
    metrics.add_row("Worst Trade", f"${e.worst_trade:+,.2f}",
                     Text("\u25bc", style="bright_red") if e.worst_trade < 0 else Text("\u00b7", style="dim"))
    metrics.add_row("Avg Hold (bars)", f"{e.avg_hold_bars:.1f}", Text("\u00b7", style="dim"))
    metrics.add_row("Win Streak", str(e.consecutive_wins),
                     Text(f"\u2191", style="bright_green") if e.consecutive_wins >= 3 else Text("\u00b7", style="dim"))
    metrics.add_row("Loss Streak", str(e.consecutive_losses),
                     Text(f"\u2193", style="bright_red") if e.consecutive_losses >= 3 else Text("\u00b7", style="dim"))

    # -- Equity curve --
    eq_title = Text("  Equity Curve", style="bold bright_white")
    eq_spark = Text("  ")
    hist = list(e.equity_history)
    if hist:
        eq_spark.append_text(render_sparkline(hist, width=60))
    else:
        eq_spark.append("\u2500" * 60, style="dim")

    # -- Risk Assessment --
    risk_section = Text()
    risk_section.append("  Risk Assessment: ", style="bold bright_white")
    dd_pct = e.max_drawdown_pct * 100
    if dd_pct < 2.0 and e.profit_factor >= 1.5 and e.sharpe_ratio >= 1.5:
        risk_section.append("LOW RISK - EXCELLENT", style="bold bright_green")
    elif dd_pct < 4.0 and e.profit_factor >= 1.0:
        risk_section.append("MODERATE RISK - ACCEPTABLE", style="bold bright_yellow")
    else:
        risk_section.append("HIGH RISK - REVIEW NEEDED", style="bold bright_red")

    content = Group(metrics, eq_title, eq_spark, risk_section)

    return Panel(
        content,
        title="[bold bright_magenta]Performance Analytics[/]",
        border_style="bright_magenta",
    )
