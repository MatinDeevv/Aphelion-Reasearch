"""
APHELION TUI — Text Area Widget (Phase 23)

A simple multi-line text buffer that the HEPHAESTUS screen can
render.  The TUI captures keystrokes; this holds the state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.panel import Panel
from rich.text import Text


@dataclass
class TextAreaState:
    """Mutable state for an editable text area."""
    lines: list[str] = field(default_factory=lambda: [""])
    cursor_row: int = 0
    cursor_col: int = 0
    max_lines: int = 200

    # ── Content access ───────────────────────────────────────────────

    @property
    def content(self) -> str:
        return "\n".join(self.lines)

    @content.setter
    def content(self, value: str) -> None:
        self.lines = value.split("\n")[:self.max_lines]
        self.cursor_row = max(0, len(self.lines) - 1)
        self.cursor_col = len(self.lines[self.cursor_row]) if self.lines else 0

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def is_empty(self) -> bool:
        return all(not line.strip() for line in self.lines)

    # ── Mutations ────────────────────────────────────────────────────

    def insert_char(self, ch: str) -> None:
        """Insert a character at cursor position."""
        if len(self.lines) == 0:
            self.lines = [""]
        row = min(self.cursor_row, len(self.lines) - 1)
        line = self.lines[row]
        col = min(self.cursor_col, len(line))
        self.lines[row] = line[:col] + ch + line[col:]
        self.cursor_col = col + len(ch)

    def insert_newline(self) -> None:
        """Insert a newline at cursor, splitting the current line."""
        if len(self.lines) >= self.max_lines:
            return
        row = min(self.cursor_row, len(self.lines) - 1)
        line = self.lines[row]
        col = min(self.cursor_col, len(line))
        self.lines[row] = line[:col]
        self.lines.insert(row + 1, line[col:])
        self.cursor_row = row + 1
        self.cursor_col = 0

    def backspace(self) -> None:
        """Delete character before cursor."""
        if self.cursor_col > 0:
            row = min(self.cursor_row, len(self.lines) - 1)
            line = self.lines[row]
            col = min(self.cursor_col, len(line))
            self.lines[row] = line[:col - 1] + line[col:]
            self.cursor_col = col - 1
        elif self.cursor_row > 0:
            # Merge with previous line
            prev = self.lines[self.cursor_row - 1]
            current = self.lines.pop(self.cursor_row)
            self.cursor_row -= 1
            self.cursor_col = len(prev)
            self.lines[self.cursor_row] = prev + current

    def clear(self) -> None:
        """Clear all content."""
        self.lines = [""]
        self.cursor_row = 0
        self.cursor_col = 0

    def select_all_and_clear(self) -> str:
        """Return current content and clear."""
        content = self.content
        self.clear()
        return content

    # ── Cursor movement ──────────────────────────────────────────────

    def cursor_up(self) -> None:
        """Move cursor up one line."""
        if self.cursor_row > 0:
            self.cursor_row -= 1
            self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))

    def cursor_down(self) -> None:
        """Move cursor down one line."""
        if self.cursor_row < len(self.lines) - 1:
            self.cursor_row += 1
            self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_row]))

    def cursor_left(self) -> None:
        """Move cursor left one character."""
        if self.cursor_col > 0:
            self.cursor_col -= 1
        elif self.cursor_row > 0:
            self.cursor_row -= 1
            self.cursor_col = len(self.lines[self.cursor_row])

    def cursor_right(self) -> None:
        """Move cursor right one character."""
        line_len = len(self.lines[self.cursor_row]) if self.lines else 0
        if self.cursor_col < line_len:
            self.cursor_col += 1
        elif self.cursor_row < len(self.lines) - 1:
            self.cursor_row += 1
            self.cursor_col = 0

    def cursor_home(self) -> None:
        """Move cursor to start of line."""
        self.cursor_col = 0

    def cursor_end(self) -> None:
        """Move cursor to end of line."""
        if self.lines:
            self.cursor_col = len(self.lines[self.cursor_row])

    def paste(self, text: str) -> None:
        """Paste multi-line text at cursor."""
        paste_lines = text.split("\n")
        if not paste_lines:
            return
        # Insert first fragment into current line
        if len(self.lines) == 0:
            self.lines = [""]
        row = min(self.cursor_row, len(self.lines) - 1)
        col = min(self.cursor_col, len(self.lines[row]))
        current = self.lines[row]
        first = current[:col] + paste_lines[0]

        if len(paste_lines) == 1:
            self.lines[row] = first + current[col:]
            self.cursor_col = len(first)
        else:
            remainder = current[col:]
            self.lines[row] = first
            for i, pl in enumerate(paste_lines[1:], 1):
                if len(self.lines) >= self.max_lines:
                    break
                if i == len(paste_lines) - 1:
                    self.lines.insert(row + i, pl + remainder)
                else:
                    self.lines.insert(row + i, pl)
            self.cursor_row = min(row + len(paste_lines) - 1, len(self.lines) - 1)
            self.cursor_col = len(paste_lines[-1])


def render_text_area(
    state: TextAreaState,
    title: str = "Code Input",
    height: int = 12,
    show_cursor: bool = True,
) -> Panel:
    """Render a text area state as a Rich Panel.

    Shows line numbers and a cursor indicator.
    """
    text = Text()
    visible_start = max(0, state.cursor_row - height + 2)
    visible_end = visible_start + height

    for i in range(visible_start, min(visible_end, len(state.lines))):
        line = state.lines[i]
        line_num = f"{i + 1:3d} "
        text.append(line_num, style="dim cyan")

        if show_cursor and i == state.cursor_row:
            col = min(state.cursor_col, len(line))
            text.append(line[:col], style="bright_white")
            if col < len(line):
                text.append(line[col], style="black on bright_yellow")
                text.append(line[col + 1:], style="bright_white")
            else:
                text.append("█", style="bright_yellow")
        else:
            text.append(line, style="bright_white")

        if i < min(visible_end, len(state.lines)) - 1:
            text.append("\n")

    # Pad remaining lines
    rendered_lines = min(visible_end, len(state.lines)) - visible_start
    for _ in range(height - rendered_lines):
        text.append("\n")
        text.append("    ~", style="dim")

    return Panel(
        text,
        title=f"[bold bright_white]{title}[/]",
        border_style="bright_cyan",
        expand=True,
    )
