"""
APHELION TUI — Equity Sparkline Widget

Renders an inline sparkline of equity history using Unicode block chars.
Supports color-gradient (green → yellow → red) based on drawdown depth.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from rich.text import Text

# Unicode block elements for 8-level resolution
_BLOCKS = " ▁▂▃▄▅▆▇█"


def render_sparkline(
    data: Sequence[float],
    width: int = 60,
    color_positive: str = "bright_green",
    color_negative: str = "bright_red",
    color_neutral: str = "bright_yellow",
) -> Text:
    """
    Return a Rich Text sparkline from *data*.

    The sparkline auto-scales to [min, max] across the visible window.
    Bars above the starting value are green; below are red.
    """
    if not data:
        return Text("─" * width, style="dim")

    # Take last `width` values
    visible = list(data)[-width:]
    lo = min(visible)
    hi = max(visible)
    span = hi - lo if hi != lo else 1.0
    start_val = visible[0]

    parts: list[tuple[str, str]] = []
    for v in visible:
        idx = int((v - lo) / span * (len(_BLOCKS) - 1))
        idx = max(0, min(len(_BLOCKS) - 1, idx))
        if v >= start_val:
            style = color_positive
        elif v < start_val * 0.99:
            style = color_negative
        else:
            style = color_neutral
        parts.append((_BLOCKS[idx], style))

    txt = Text()
    for ch, style in parts:
        txt.append(ch, style=style)
    return txt


def render_confidence_sparkline(
    data: Sequence[float],
    width: int = 40,
) -> Text:
    """Sparkline for HYDRA confidence history [0-1]."""
    if not data:
        return Text("─" * width, style="dim")
    visible = list(data)[-width:]
    txt = Text()
    for v in visible:
        idx = int(v * (len(_BLOCKS) - 1))
        idx = max(0, min(len(_BLOCKS) - 1, idx))
        if v >= 0.7:
            style = "bright_green"
        elif v >= 0.4:
            style = "bright_yellow"
        else:
            style = "bright_red"
        txt.append(_BLOCKS[idx], style=style)
    return txt
