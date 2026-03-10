"""
APHELION TUI — Form Field Widget (Phase 23)

Interactive form input fields for the Setup screen.
Each field holds a label, value, type info, and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from rich.table import Table
from rich.text import Text


@dataclass
class FormField:
    """A single form field with label, value, and metadata."""
    key: str             # config key path (e.g. "trading.symbol")
    label: str           # display label
    value: str = ""      # current string value
    field_type: str = "text"   # text | number | password | select
    options: list[str] = field(default_factory=list)  # for select type
    placeholder: str = ""
    validation_error: str = ""
    section: str = ""    # section grouping

    @property
    def display_value(self) -> str:
        """Value to display — masked for passwords."""
        if self.field_type == "password" and self.value:
            return "•" * min(len(self.value), 12)
        return self.value or self.placeholder

    def set_value(self, val: str) -> None:
        """Set value and clear validation error."""
        self.value = val
        self.validation_error = ""

    def validate(self) -> bool:
        """Run basic validation. Returns True if valid."""
        if self.field_type == "number":
            try:
                float(self.value) if self.value else None
            except ValueError:
                self.validation_error = "Must be a number"
                return False
        if self.field_type == "select" and self.options:
            if self.value and self.value not in self.options:
                self.validation_error = f"Must be one of: {', '.join(self.options)}"
                return False
        self.validation_error = ""
        return True


@dataclass
class FormState:
    """State for a form with multiple fields."""
    fields: list[FormField] = field(default_factory=list)
    active_index: int = 0
    dirty: bool = False   # True if any field modified since last save

    @property
    def active_field(self) -> Optional[FormField]:
        if 0 <= self.active_index < len(self.fields):
            return self.fields[self.active_index]
        return None

    def next_field(self) -> None:
        """Move to next field."""
        if self.fields:
            self.active_index = (self.active_index + 1) % len(self.fields)

    def prev_field(self) -> None:
        """Move to previous field."""
        if self.fields:
            self.active_index = (self.active_index - 1) % len(self.fields)

    def next_section(self) -> None:
        """Move to first field of next section."""
        if not self.fields:
            return
        current_section = self.fields[self.active_index].section
        # Find first field of next section
        for i in range(self.active_index + 1, len(self.fields)):
            if self.fields[i].section != current_section:
                self.active_index = i
                return
        # Wrap to first section
        if self.fields[0].section != current_section:
            self.active_index = 0

    def prev_section(self) -> None:
        """Move to first field of previous section."""
        if not self.fields:
            return
        current_section = self.fields[self.active_index].section
        # Find last field before current section, then find section start
        prev_idx = -1
        for i in range(self.active_index - 1, -1, -1):
            if self.fields[i].section != current_section:
                prev_idx = i
                break
        if prev_idx >= 0:
            # Now find start of that section
            target_section = self.fields[prev_idx].section
            for i in range(prev_idx, -1, -1):
                if self.fields[i].section != target_section:
                    self.active_index = i + 1
                    return
            self.active_index = 0
        else:
            # Wrap to last section
            last_section = self.fields[-1].section
            for i in range(len(self.fields) - 1, -1, -1):
                if self.fields[i].section != last_section:
                    self.active_index = i + 1
                    return
            self.active_index = 0

    def get_value(self, key: str) -> str:
        """Get a field value by key."""
        for f in self.fields:
            if f.key == key:
                return f.value
        return ""

    def set_value(self, key: str, value: str) -> None:
        """Set a field value by key."""
        for f in self.fields:
            if f.key == key:
                f.set_value(value)
                self.dirty = True
                return

    def validate_all(self) -> list[str]:
        """Validate all fields. Returns list of error messages."""
        errors = []
        for f in self.fields:
            if not f.validate():
                errors.append(f"{f.label}: {f.validation_error}")
        return errors

    def to_dict(self) -> dict[str, str]:
        """Export all field values as a flat dict."""
        return {f.key: f.value for f in self.fields}

    @property
    def sections(self) -> list[str]:
        """Unique section names in order."""
        seen: list[str] = []
        for f in self.fields:
            if f.section and f.section not in seen:
                seen.append(f.section)
        return seen


def render_form(
    state: FormState,
    title: str = "Settings",
    width: int = 60,
) -> Table:
    """Render a form as a Rich Table.

    The active field is highlighted. Validation errors shown in red.
    """
    table = Table(
        show_header=False,
        expand=True,
        show_edge=False,
        padding=(0, 1),
    )
    table.add_column("label", ratio=1)
    table.add_column("value", ratio=2)

    current_section = ""

    for i, fld in enumerate(state.fields):
        # Section header
        if fld.section and fld.section != current_section:
            current_section = fld.section
            section_text = Text(f"\n  {current_section.upper()}", style="bold bright_yellow")
            divider = Text("  " + "─" * 40, style="dim")
            table.add_row(section_text, Text(""))
            table.add_row(divider, Text(""))

        is_active = (i == state.active_index)
        label_style = "bold bright_white" if is_active else "bright_white"
        label = Text(f"  {'▶' if is_active else ' '} {fld.label}", style=label_style)

        value_text = Text()
        display = fld.display_value
        if is_active:
            value_text.append("[ ", style="bright_cyan")
            value_text.append(display or " ", style="bold bright_white")
            value_text.append(" ]", style="bright_cyan")
        else:
            value_text.append("[ ", style="dim")
            value_text.append(display or " ", style="bright_white")
            value_text.append(" ]", style="dim")

        if fld.validation_error:
            value_text.append(f"  ⚠ {fld.validation_error}", style="bright_red")

        table.add_row(label, value_text)

    return table
