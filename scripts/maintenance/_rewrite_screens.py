"""Script to rewrite TUI screen files with Bloomberg-grade implementations."""
import os

BASE = r"c:\Users\marti\Aphelion\aphelion\tui\screens"

# ═══════════════════════════════════════════════════════════════════════
# equity.py
# ═══════════════════════════════════════════════════════════════════════
equity_py = '''\
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
        spark_line.append("\\u2500" * 50, style="dim")

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
'''

# ═══════════════════════════════════════════════════════════════════════
# positions.py
# ═══════════════════════════════════════════════════════════════════════
positions_py = '''\
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
            "\\u2014", "\\u2014", "\\u2014", "\\u2014", "\\u2014", "\\u2014", "\\u2014", "\\u2014",
            Text("No open positions", style="dim"), "\\u2014", "\\u2014",
        )
    else:
        for p in state.positions:
            dir_icon = "\\u25b2" if p.direction == "LONG" else "\\u25bc"
            dir_style = "bold bright_green" if p.direction == "LONG" else "bold bright_red"
            pnl_style = "bold bright_green" if p.unrealized_pnl >= 0 else "bold bright_red"
            pnl_str = f"${p.unrealized_pnl:+,.2f}"

            rr_str = f"{p.risk_reward:.1f}" if p.risk_reward > 0 else "\\u2014"
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
                str(p.hold_bars) if p.hold_bars > 0 else "\\u2014",
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
'''

# ═══════════════════════════════════════════════════════════════════════
# event_log.py
# ═══════════════════════════════════════════════════════════════════════
event_log_py = '''\
"""APHELION TUI - Event Log Panel  (v2 - Bloomberg-grade)

Color-coded scrolling event log with level icons, timestamps,
and message type indicators.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState

_LEVEL_STYLE = {
    "INFO": "white",
    "WARNING": "bold bright_yellow",
    "ERROR": "bold bright_red",
    "FILL": "bold bright_green",
    "REJECT": "bold bright_red",
    "SENTINEL": "bold bright_magenta",
    "HYDRA": "bold bright_yellow",
    "SYSTEM": "bold bright_cyan",
    "TRADE": "bold bright_green",
}

_LEVEL_ICON = {
    "INFO": "\\u00b7",
    "WARNING": "\\u26a0",
    "ERROR": "\\u2717",
    "FILL": "\\u2713",
    "REJECT": "\\u2297",
    "SENTINEL": "\\u25c8",
    "HYDRA": "\\u26a1",
    "SYSTEM": "\\u25c6",
    "TRADE": "\\u25c9",
}


def build_log_panel(state: TUIState, max_visible: int = 20) -> Panel:
    """Build the scrolling event-log panel (Bloomberg-grade)."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(max_width=9)
    table.add_column(max_width=2)
    table.add_column(max_width=9)
    table.add_column(ratio=1)

    visible = state.log[-max_visible:] if state.log else []

    if not visible:
        table.add_row(
            "", "",
            Text("IDLE", style="dim"),
            Text("Waiting for events...", style="dim"),
        )
    else:
        for entry in visible:
            ts = entry.timestamp.strftime("%H:%M:%S")
            level_style = _LEVEL_STYLE.get(entry.level, "white")
            icon = _LEVEL_ICON.get(entry.level, "\\u00b7")

            badge = Text(f" {entry.level:8s}", style=f"{level_style}")

            table.add_row(
                Text(ts, style="dim rgb(140,140,180)"),
                Text(icon, style=level_style),
                badge,
                Text(entry.message, style=level_style, overflow="ellipsis"),
            )

    summary = Text()
    summary.append(f"  Total: {len(state.log)} events", style="dim")
    fills = sum(1 for e in state.log if e.level == "FILL")
    errors = sum(1 for e in state.log if e.level == "ERROR")
    rejects = sum(1 for e in state.log if e.level == "REJECT")
    if fills:
        summary.append(f"  Fills:{fills}", style="bright_green")
    if errors:
        summary.append(f"  Errors:{errors}", style="bright_red")
    if rejects:
        summary.append(f"  Rejects:{rejects}", style="bright_red")

    content = Group(table, summary)

    return Panel(
        content,
        title="[bold rgb(140,140,180)]Event Log[/]",
        border_style="rgb(60,60,100)",
    )
'''

# Write all files
files = {
    os.path.join(BASE, "equity.py"): equity_py,
    os.path.join(BASE, "positions.py"): positions_py,
    os.path.join(BASE, "event_log.py"): event_log_py,
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {path}")

print("Done!")
