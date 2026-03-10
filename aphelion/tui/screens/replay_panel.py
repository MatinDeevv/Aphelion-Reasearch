"""
APHELION TUI — Replay Panel  (v2 — Bloomberg-grade)

Displays trade replay / review mode: scrollable trade history,
entry/exit details, feature snapshot at time of trade.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState


def build_replay_panel(state: TUIState) -> Panel:
    """Build the trade replay panel."""

    # Replay data from extended state
    replay_trades = getattr(state, "replay_trades", [])
    selected_idx = getattr(state, "replay_selected_idx", 0)
    replay_mode = getattr(state, "replay_mode", False)

    # ── Mode banner ──
    banner = Text()
    if replay_mode:
        banner.append("  ▶ REPLAY MODE  ", style="bold bright_cyan on rgb(0,30,50)")
        banner.append(f"  Trade {selected_idx + 1}/{len(replay_trades)}", style="dim")
    else:
        banner.append("  ○ REPLAY INACTIVE  ", style="dim")

    # ── Trade list ──
    trade_table = Table(show_header=True, expand=True, show_edge=False, padding=(0, 1))
    trade_table.add_column("#", width=4, style="dim")
    trade_table.add_column("Dir", width=5)
    trade_table.add_column("Entry", justify="right")
    trade_table.add_column("Exit", justify="right")
    trade_table.add_column("PnL", justify="right")
    trade_table.add_column("R:R", justify="right")
    trade_table.add_column("Time", style="dim")

    for i, t in enumerate(replay_trades[-20:]):
        direction = getattr(t, "direction", "?")
        entry = getattr(t, "entry_price", 0.0)
        exit_p = getattr(t, "exit_price", 0.0)
        pnl = getattr(t, "pnl", 0.0)
        rr = getattr(t, "risk_reward", 0.0)
        time_str = getattr(t, "time_str", "")

        pnl_style = "bright_green" if pnl >= 0 else "bright_red"
        row_style = "bold" if i == selected_idx else ""

        trade_table.add_row(
            str(i + 1),
            direction,
            f"{entry:.2f}",
            f"{exit_p:.2f}",
            Text(f"${pnl:+.2f}", style=pnl_style),
            f"{rr:.1f}R",
            time_str,
            style=row_style,
        )

    if not replay_trades:
        trade_table.add_row("—", "—", "—", "—", Text("No trades", style="dim"), "—", "—")

    # ── Selected trade detail ──
    detail = Text()
    if replay_trades and 0 <= selected_idx < len(replay_trades):
        t = replay_trades[selected_idx]
        detail.append("\n  Trade Detail:\n", style="bright_white bold")
        detail.append(f"    Direction:  {getattr(t, 'direction', '?')}\n", style="bright_white")
        detail.append(f"    Entry:      {getattr(t, 'entry_price', 0):.2f}\n", style="bright_white")
        detail.append(f"    Exit:       {getattr(t, 'exit_price', 0):.2f}\n", style="bright_white")
        detail.append(f"    Stop Loss:  {getattr(t, 'stop_loss', 0):.2f}\n", style="bright_white")
        detail.append(f"    Take Profit:{getattr(t, 'take_profit', 0):.2f}\n", style="bright_white")
        pnl = getattr(t, 'pnl', 0.0)
        pstyle = "bright_green" if pnl >= 0 else "bright_red"
        detail.append(f"    PnL:        ", style="bright_white")
        detail.append(f"${pnl:+.2f}\n", style=pstyle)
        # Feature snapshot if available
        features = getattr(t, "features", {})
        if features:
            detail.append("    Features:\n", style="dim")
            for k, v in list(features.items())[:8]:
                detail.append(f"      {k}: {v}\n", style="dim")

    content = Group(banner, Text(""), trade_table, detail)
    return Panel(
        content,
        title="[bold bright_white]TRADE REPLAY[/]",
        border_style="bright_cyan",
        expand=True,
    )
