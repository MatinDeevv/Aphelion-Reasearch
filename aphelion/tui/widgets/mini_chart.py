"""
APHELION TUI — Mini Chart Widget

Renders a mini ASCII chart from price history using plotext (if available)
or a fallback Unicode sparkline.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from rich.text import Text

_BLOCKS = " ▁▂▃▄▅▆▇█"


def render_mini_chart(
    data: Sequence[float],
    width: int = 40,
    height: int = 6,
    title: str = "",
) -> str:
    """
    Render a mini ASCII line chart.

    Returns a multi-line string suitable for Rich rendering.
    Uses a simple braille/block-char approach for terminal compatibility.
    """
    if not data or len(data) < 2:
        return "  No data yet" + ("\n" * (height - 1))

    visible = list(data)[-width:]
    lo = min(visible)
    hi = max(visible)
    span = hi - lo if hi != lo else 1.0

    # Build grid
    grid = [[" " for _ in range(len(visible))] for _ in range(height)]
    for col, val in enumerate(visible):
        row = int((val - lo) / span * (height - 1))
        row = max(0, min(height - 1, row))
        grid[height - 1 - row][col] = "█"
        # Fill below for area chart effect
        for r in range(height - row, height):
            if grid[r][col] == " ":
                grid[r][col] = "░"

    lines = []
    for row_idx, row in enumerate(grid):
        # Y-axis labels
        if row_idx == 0:
            label = f"{hi:>8.2f}│"
        elif row_idx == height - 1:
            label = f"{lo:>8.2f}│"
        elif row_idx == height // 2:
            mid = (hi + lo) / 2
            label = f"{mid:>8.2f}│"
        else:
            label = "        │"
        lines.append(label + "".join(row))

    # X-axis
    lines.append("        └" + "─" * len(visible))
    if title:
        lines.insert(0, f"  {title}")

    return "\n".join(lines)


def render_ohlc_bars(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    width: int = 40,
    height: int = 8,
) -> str:
    """
    Render mini OHLC bar chart using Unicode block characters.

    Each bar uses │ for the wick and █/▄/▀ for the body.
    Green for bullish, red for bearish.
    """
    if not closes or len(closes) < 2:
        return "  No OHLC data"

    n = min(width, len(closes))
    o = list(opens)[-n:]
    h = list(highs)[-n:]
    l = list(lows)[-n:]
    c = list(closes)[-n:]

    all_vals = h + l
    lo = min(all_vals)
    hi = max(all_vals)
    span = hi - lo if hi != lo else 1.0

    def scale(v: float) -> int:
        return int((v - lo) / span * (height - 1))

    grid = [[" " for _ in range(n)] for _ in range(height)]

    for col in range(n):
        h_row = height - 1 - scale(h[col])
        l_row = height - 1 - scale(l[col])
        o_row = height - 1 - scale(o[col])
        c_row = height - 1 - scale(c[col])
        body_top = min(o_row, c_row)
        body_bot = max(o_row, c_row)

        # Wick
        for r in range(h_row, l_row + 1):
            grid[r][col] = "│"
        # Body
        for r in range(body_top, body_bot + 1):
            grid[r][col] = "█" if c[col] >= o[col] else "▒"

    lines = []
    for row_idx, row in enumerate(grid):
        if row_idx == 0:
            label = f"{hi:>8.2f}│"
        elif row_idx == height - 1:
            label = f"{lo:>8.2f}│"
        else:
            label = "        │"
        lines.append(label + "".join(row))
    lines.append("        └" + "─" * n)

    return "\n".join(lines)
