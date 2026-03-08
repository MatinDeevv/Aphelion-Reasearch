"""
APHELION TUI — Header Bar
Session name, clock, market status, bars processed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState


def build_header(state: TUIState) -> Panel:
    """3-row header showing session info and market status."""
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1, justify="center")
    table.add_column(ratio=1, justify="right")

    # Left: session name
    session_text = Text(f"  {state.session_name}", style="bold cyan")

    # Center: current time + session
    now = state.current_time or datetime.now(timezone.utc)
    time_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    session_label = state.current_session or "DEAD_ZONE"
    center = Text(f"{time_str}  [{session_label}]", style="white")

    # Right: market status + bar count
    if state.market_open:
        market_text = Text("● MARKET OPEN", style="bold green")
    else:
        market_text = Text("○ MARKET CLOSED", style="dim red")
    bars_text = Text(f"  Bars: {state.bars_processed}", style="dim")
    right = market_text + bars_text

    table.add_row(session_text, center, right)

    return Panel(
        table,
        title="[bold white]◈ APHELION TRADING SYSTEM ◈[/]",
        border_style="bright_blue",
    )
