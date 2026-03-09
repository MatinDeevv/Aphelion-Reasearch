"""
APHELION TUI — Risk Gauge Widget

Renders horizontal bar gauges with color-coded thresholds.
Bloomberg-style: green → amber → red zones.
"""

from __future__ import annotations

from rich.text import Text


def render_gauge(
    value: float,
    maximum: float,
    width: int = 20,
    label: str = "",
    warn_pct: float = 0.6,
    crit_pct: float = 0.8,
    invert: bool = False,
) -> Text:
    """
    Horizontal bar gauge.

    Parameters
    ----------
    value   : current value
    maximum : maximum scale value
    width   : character width of the bar
    label   : optional label prefix
    warn_pct: fraction of max where colour turns amber
    crit_pct: fraction of max where colour turns red
    invert  : if True, lower is worse (e.g. confidence)
    """
    if maximum <= 0:
        frac = 0.0
    else:
        frac = min(value / maximum, 1.0)

    filled = int(frac * width)
    empty = width - filled

    # Determine colour
    if invert:
        if frac >= crit_pct:
            color = "bright_green"
        elif frac >= warn_pct:
            color = "bright_yellow"
        else:
            color = "bright_red"
    else:
        if frac >= crit_pct:
            color = "bright_red"
        elif frac >= warn_pct:
            color = "bright_yellow"
        else:
            color = "bright_green"

    bar_char = "█"
    empty_char = "░"

    txt = Text()
    if label:
        txt.append(f"{label} ", style="bold")
    txt.append(bar_char * filled, style=color)
    txt.append(empty_char * empty, style="dim")
    txt.append(f" {value:.1f}/{maximum:.1f}", style="white")
    return txt


def render_breaker_indicator(
    label: str,
    triggered: bool,
    width: int = 12,
) -> Text:
    """
    Circuit breaker status indicator.

    ● LABEL ████████████  (red if triggered, green if clear)
    """
    txt = Text()
    if triggered:
        txt.append("● ", style="bold bright_red blink")
        txt.append(f"{label} ", style="bold bright_red")
        txt.append("█" * width, style="bright_red")
        txt.append(" TRIPPED", style="bold bright_red")
    else:
        txt.append("○ ", style="bright_green")
        txt.append(f"{label} ", style="bright_green")
        txt.append("░" * width, style="dim green")
        txt.append(" CLEAR", style="dim green")
    return txt


def render_mini_bar(
    value: float,
    maximum: float,
    width: int = 10,
) -> Text:
    """Compact inline bar without label."""
    if maximum <= 0:
        frac = 0.0
    else:
        frac = min(value / maximum, 1.0)
    filled = int(frac * width)
    empty = width - filled
    color = "bright_green" if frac < 0.5 else ("bright_yellow" if frac < 0.8 else "bright_red")
    txt = Text()
    txt.append("█" * filled, style=color)
    txt.append("░" * empty, style="dim")
    return txt
