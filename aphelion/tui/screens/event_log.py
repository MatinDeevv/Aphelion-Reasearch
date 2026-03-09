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
    "INFO": "\u00b7",
    "WARNING": "\u26a0",
    "ERROR": "\u2717",
    "FILL": "\u2713",
    "REJECT": "\u2297",
    "SENTINEL": "\u25c8",
    "HYDRA": "\u26a1",
    "SYSTEM": "\u25c6",
    "TRADE": "\u25c9",
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
            icon = _LEVEL_ICON.get(entry.level, "\u00b7")

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
