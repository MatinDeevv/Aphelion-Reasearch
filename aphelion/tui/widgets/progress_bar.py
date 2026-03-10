"""
APHELION TUI — Progress Bar Widget (Phase 23)

Animated progress bar for forge, training, and backtest operations.
"""

from __future__ import annotations

from rich.text import Text


def render_progress_bar(
    label: str,
    percent: float,
    message: str = "",
    width: int = 30,
    elapsed: float = 0.0,
    style_fill: str = "bright_green",
    style_empty: str = "dim",
) -> Text:
    """Render a horizontal progress bar with label and message.

    Parameters
    ----------
    label    : prefix label (e.g. "▶ Parsing...")
    percent  : 0.0 to 1.0
    message  : detail text after the bar
    width    : character width of bar
    elapsed  : elapsed seconds for time display
    """
    pct = max(0.0, min(1.0, percent))
    filled = int(pct * width)
    empty = width - filled

    txt = Text()
    txt.append(f"  {label}  ", style="bold bright_white")
    txt.append("[", style="dim")
    txt.append("█" * filled, style=style_fill)
    txt.append("░" * empty, style=style_empty)
    txt.append("]", style="dim")
    txt.append(f"  {pct * 100:4.0f}%", style="bright_cyan")

    if elapsed > 0:
        mins, secs = divmod(int(elapsed), 60)
        txt.append(f"   {mins:02d}:{secs:02d}", style="dim")

    if message:
        txt.append(f"\n    → {message}", style="bright_yellow")

    return txt


def render_epoch_progress(
    current: int,
    total: int,
    width: int = 30,
    elapsed: float = 0.0,
) -> Text:
    """Render an epoch-based progress bar for training."""
    pct = current / total if total > 0 else 0.0
    label = f"Epoch {current}/{total}"
    return render_progress_bar(
        label=label,
        percent=pct,
        width=width,
        elapsed=elapsed,
    )


def render_loss_sparkline(history: list[float], width: int = 40) -> Text:
    """Render a mini loss-curve sparkline from history."""
    if not history:
        return Text("  Loss curve: (waiting for data)", style="dim")

    blocks = "▁▂▃▄▅▆▇█"
    mn = min(history)
    mx = max(history)
    rng = mx - mn if mx > mn else 1.0

    # Downsample to width
    step = max(1, len(history) // width)
    samples = history[::step][:width]

    txt = Text()
    txt.append("  Loss curve: ", style="dim")
    for v in samples:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        txt.append(blocks[idx], style="bright_cyan")

    # Trend indicator
    if len(history) >= 5:
        recent = sum(history[-5:]) / 5
        earlier = sum(history[:5]) / 5
        if recent < earlier * 0.95:
            txt.append(" (converging)", style="bright_green")
        elif recent > earlier * 1.05:
            txt.append(" (diverging)", style="bright_red")
        else:
            txt.append(" (plateau)", style="bright_yellow")

    return txt
