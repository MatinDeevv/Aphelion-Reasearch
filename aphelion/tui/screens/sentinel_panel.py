"""
APHELION TUI — SENTINEL Risk Panel
Shows circuit-breaker status, drawdown, position limits, exposure.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState


def _status_dot(ok: bool, invert: bool = False) -> Text:
    """Green dot for OK, red dot for triggered."""
    if invert:
        ok = not ok
    if ok:
        return Text("●", style="bold green")
    return Text("●", style="bold red")


def _breaker_row(label: str, triggered: bool) -> Text:
    """Format a circuit breaker row."""
    t = Text("  ")
    if triggered:
        t.append("■ ", style="bold red")
        t.append(f"{label}: TRIGGERED", style="bold red")
    else:
        t.append("□ ", style="green")
        t.append(f"{label}: OK", style="green")
    return t


def build_sentinel_panel(state: TUIState) -> Panel:
    """Build the SENTINEL risk status panel."""
    s = state.sentinel

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column()

    # Trading allowed
    if s.trading_allowed:
        status_line = Text("  Trading: ", style="white")
        status_line.append("ALLOWED", style="bold green")
    else:
        status_line = Text("  Trading: ", style="white")
        status_line.append("BLOCKED", style="bold red blink")
    table.add_row(status_line)

    # Circuit breakers
    table.add_row(_breaker_row("L1 — Cooldown", s.l1_triggered))
    table.add_row(_breaker_row("L2 — Daily Loss Limit", s.l2_triggered))
    table.add_row(_breaker_row("L3 — Emergency Disconnect", s.l3_triggered))

    # Drawdown
    dd_pct = s.daily_drawdown_pct * 100
    dd_style = "green" if dd_pct < 2.0 else ("yellow" if dd_pct < 4.0 else "red")
    dd_line = Text(f"  Daily Drawdown: {dd_pct:.2f}%", style=dd_style)
    table.add_row(dd_line)

    # Position count
    pos_line = Text(
        f"  Positions: {s.open_positions}/{s.max_positions}",
        style="white",
    )
    table.add_row(pos_line)

    # Exposure
    exp_pct = s.total_exposure_pct * 100
    max_exp_pct = s.max_exposure_pct * 100
    exp_style = "green" if exp_pct < max_exp_pct * 0.8 else "yellow"
    exp_line = Text(f"  Exposure: {exp_pct:.1f}% / {max_exp_pct:.1f}%", style=exp_style)
    table.add_row(exp_line)

    # Peak equity
    peak_line = Text(
        f"  Session Peak: ${s.session_peak_equity:,.2f}", style="dim"
    )
    table.add_row(peak_line)

    return Panel(
        table,
        title="[bold red]🛡 SENTINEL Risk[/]",
        border_style="red",
    )
