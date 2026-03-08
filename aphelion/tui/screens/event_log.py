"""
APHELION TUI — Event Log Panel
Shows the most recent fills, rejections, and system events.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState

_LEVEL_STYLE = {
    "INFO": "white",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "FILL": "bold green",
    "REJECT": "bold red",
    "SENTINEL": "magenta",
}


def build_log_panel(state: TUIState) -> Panel:
    """Build the scrolling event-log panel."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(max_width=8)    # time
    table.add_column(max_width=8)    # level
    table.add_column(ratio=1)        # message

    # Show most recent 15 entries (bottom = newest)
    visible = state.log[-15:] if state.log else []

    if not visible:
        table.add_row("", "", Text("Waiting for events…", style="dim"))
    else:
        for entry in visible:
            ts = entry.timestamp.strftime("%H:%M:%S")
            level_style = _LEVEL_STYLE.get(entry.level, "white")
            table.add_row(
                Text(ts, style="dim"),
                Text(entry.level, style=level_style),
                Text(entry.message, style=level_style, overflow="ellipsis"),
            )

    return Panel(
        table,
        title="[bold white]📋 Event Log[/]",
        border_style="bright_black",
    )
