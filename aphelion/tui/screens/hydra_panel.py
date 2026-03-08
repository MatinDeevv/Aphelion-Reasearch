"""
APHELION TUI — HYDRA Signal Panel
Shows direction, confidence, horizon agreement, gate weights, top features.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState

_DIR_STYLE = {
    "LONG": "bold green",
    "SHORT": "bold red",
    "FLAT": "dim yellow",
}

_SUB_MODEL_NAMES = ["TFT", "LSTM", "CNN", "MoE"]
_MOE_EXPERT_NAMES = ["TREND", "RANGE", "VOL_EXP", "NEWS"]


def _bar(value: float, width: int = 20) -> str:
    """Render a horizontal bar: ████░░░░░░ 65%"""
    filled = int(round(value * width))
    empty = width - filled
    return "█" * filled + "░" * empty


def _prob_row(label: str, probs: list[float]) -> Text:
    """Format [P(SHORT), P(FLAT), P(LONG)] as a colored line."""
    short_pct = probs[0] * 100
    flat_pct = probs[1] * 100
    long_pct = probs[2] * 100
    t = Text(f"  {label:>4s}: ")
    t.append(f"S {short_pct:4.1f}% ", style="red")
    t.append(f"F {flat_pct:4.1f}% ", style="yellow")
    t.append(f"L {long_pct:4.1f}%", style="green")
    return t


def build_hydra_panel(state: TUIState) -> Panel:
    """Build the HYDRA signal display panel."""
    h = state.hydra

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column()

    # Direction + confidence
    dir_style = _DIR_STYLE.get(h.direction, "white")
    dir_line = Text()
    dir_line.append(f"  Direction: ", style="white")
    dir_line.append(f"{h.direction}", style=dir_style)
    dir_line.append(f"  Conf: {h.confidence:.1%}", style="white")
    dir_line.append(f"  Unc: {h.uncertainty:.3f}", style="dim")
    table.add_row(dir_line)

    # Horizon agreement
    agree_line = Text(f"  Horizon Agreement: {h.horizon_agreement:.0%}  ")
    agree_line.append(_bar(h.horizon_agreement, 15))
    table.add_row(agree_line)

    # Per-horizon probabilities
    table.add_row(_prob_row("5m", h.probs_5m))
    table.add_row(_prob_row("15m", h.probs_15m))
    table.add_row(_prob_row("1h", h.probs_1h))

    # Gate attention weights
    gate_line = Text("  Gate: ")
    for name, w in zip(_SUB_MODEL_NAMES, h.gate_weights):
        gate_line.append(f"{name}={w:.0%} ", style="cyan")
    table.add_row(gate_line)

    # MoE routing
    moe_line = Text("  MoE:  ")
    for name, w in zip(_MOE_EXPERT_NAMES, h.moe_routing):
        moe_line.append(f"{name}={w:.0%} ", style="magenta")
    table.add_row(moe_line)

    # Top features (up to 5)
    if h.top_features:
        feat_parts = [f"{n}={v:.2f}" for n, v in h.top_features[:5]]
        feat_line = Text("  Top: " + "  ".join(feat_parts), style="dim")
        table.add_row(feat_line)

    return Panel(
        table,
        title="[bold yellow]⚡ HYDRA Signal[/]",
        border_style="yellow",
    )
