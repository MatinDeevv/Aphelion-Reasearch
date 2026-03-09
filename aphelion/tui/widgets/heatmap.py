"""
APHELION TUI — Feature Heatmap Widget

Renders a colour-coded grid of feature importance / gate weights.
Bloomberg-style information density.
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text


_HEAT_PALETTE = [
    "rgb(0,0,80)",      # cold - deep blue
    "rgb(0,50,150)",     # cool blue
    "rgb(0,128,128)",    # teal
    "rgb(0,180,80)",     # green
    "rgb(128,200,0)",    # lime
    "rgb(200,200,0)",    # yellow
    "rgb(255,160,0)",    # orange
    "rgb(255,80,0)",     # hot orange
    "rgb(255,0,0)",      # red
    "rgb(255,0,80)",     # hot red
]


def _heat_color(value: float, lo: float = 0.0, hi: float = 1.0) -> str:
    """Map a value to a heatmap colour string."""
    if hi <= lo:
        idx = 0
    else:
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        idx = int(frac * (len(_HEAT_PALETTE) - 1))
    return _HEAT_PALETTE[idx]


def render_feature_heatmap(
    features: list[tuple[str, float]],
    max_features: int = 8,
) -> Table:
    """
    Render top features as a heatmap table.

    Each row: feature name │ coloured bar │ importance value.
    """
    table = Table(
        show_header=True,
        header_style="bold bright_cyan",
        expand=True,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Feature", style="bright_white", ratio=2)
    table.add_column("Importance", ratio=3)
    table.add_column("Value", justify="right", style="bright_white", ratio=1)

    if not features:
        table.add_row("—", Text("No data", style="dim"), "—")
        return table

    # Sort descending, take top N
    sorted_feats = sorted(features, key=lambda x: abs(x[1]), reverse=True)[:max_features]
    hi = max(abs(f[1]) for f in sorted_feats) if sorted_feats else 1.0

    for name, val in sorted_feats:
        abs_val = abs(val)
        width = int((abs_val / hi) * 16) if hi > 0 else 0
        color = _heat_color(abs_val, 0.0, hi)
        bar = Text("█" * width + "░" * (16 - width), style=color)
        table.add_row(name, bar, f"{val:+.4f}")

    return table


def render_gate_weights(
    weights: list[float],
    labels: list[str] | None = None,
) -> Table:
    """Render HYDRA gate weights as coloured horizontal bars."""
    if labels is None:
        labels = [f"Expert-{i}" for i in range(len(weights))]

    table = Table(
        show_header=False,
        expand=True,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Expert", ratio=2)
    table.add_column("Weight", ratio=3)
    table.add_column("Pct", justify="right", ratio=1)

    hi = max(weights) if weights else 1.0
    for lbl, w in zip(labels, weights):
        width = int((w / hi) * 14) if hi > 0 else 0
        color = _heat_color(w, 0.0, hi)
        bar = Text("█" * width + "░" * (14 - width), style=color)
        table.add_row(
            Text(lbl, style="bright_white"),
            bar,
            f"{w * 100:.1f}%",
        )
    return table
