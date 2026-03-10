"""
APHELION TUI — File Picker Widget (Phase 23)

Simple file browser for selecting checkpoints, data files, etc.
Renders a list of files/directories from a given path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from rich.panel import Panel
from rich.text import Text


@dataclass
class FilePickerState:
    """State for a file picker dialog."""
    current_dir: str = "."
    entries: list[str] = field(default_factory=list)
    selected_index: int = 0
    filter_ext: str = ""   # e.g. ".pt", ".csv", ".parquet"
    selected_path: Optional[str] = None
    visible: bool = False

    def open(self, start_dir: str = ".", filter_ext: str = "") -> None:
        """Open the file picker at a directory."""
        self.visible = True
        self.filter_ext = filter_ext
        self.selected_path = None
        self.navigate_to(start_dir)

    def navigate_to(self, path: str) -> None:
        """Navigate to a directory and list contents."""
        try:
            path = os.path.abspath(path)
            self.current_dir = path
            raw = sorted(os.listdir(path))
            entries = [".."]  # parent directory

            for name in raw:
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    entries.append(name + "/")
                elif not self.filter_ext or name.endswith(self.filter_ext):
                    entries.append(name)

            self.entries = entries
            self.selected_index = 0
        except OSError:
            self.entries = [".."]
            self.selected_index = 0

    def move_up(self) -> None:
        if self.selected_index > 0:
            self.selected_index -= 1

    def move_down(self) -> None:
        if self.selected_index < len(self.entries) - 1:
            self.selected_index += 1

    def select(self) -> Optional[str]:
        """Select the current entry. Returns path or None if navigating."""
        if not self.entries:
            return None

        entry = self.entries[self.selected_index]

        if entry == "..":
            parent = os.path.dirname(self.current_dir)
            self.navigate_to(parent)
            return None

        full_path = os.path.join(self.current_dir, entry.rstrip("/"))

        if entry.endswith("/"):
            self.navigate_to(full_path)
            return None

        self.selected_path = full_path
        self.visible = False
        return full_path

    def cancel(self) -> None:
        """Close without selection."""
        self.visible = False
        self.selected_path = None


def render_file_picker(state: FilePickerState, height: int = 15) -> Panel:
    """Render the file picker as a Rich Panel."""
    text = Text()
    text.append(f"  {state.current_dir}\n", style="bold bright_yellow")
    text.append("  " + "─" * 50 + "\n", style="dim")

    visible_start = max(0, state.selected_index - height + 3)
    visible_end = visible_start + height - 2

    for i in range(visible_start, min(visible_end, len(state.entries))):
        entry = state.entries[i]
        is_selected = (i == state.selected_index)
        prefix = "▶ " if is_selected else "  "

        if entry.endswith("/"):
            style = "bold bright_cyan" if is_selected else "bright_cyan"
            text.append(f"  {prefix}📁 {entry}\n", style=style)
        elif entry == "..":
            style = "bold bright_yellow" if is_selected else "bright_yellow"
            text.append(f"  {prefix}⬆  {entry}\n", style=style)
        else:
            style = "bold bright_white" if is_selected else "bright_white"
            text.append(f"  {prefix}📄 {entry}\n", style=style)

    # Pad
    rendered = min(visible_end, len(state.entries)) - visible_start
    for _ in range(height - 2 - rendered):
        text.append("\n")

    text.append("\n  [ENTER] Select   [ESC] Cancel", style="dim")

    return Panel(
        text,
        title="[bold bright_white]Select File[/]",
        border_style="bright_cyan",
        expand=False,
        width=60,
    )
