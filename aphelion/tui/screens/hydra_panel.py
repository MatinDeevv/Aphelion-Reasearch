"""
APHELION TUI — HYDRA Signal Panel  (v2 — Bloomberg-grade)

Ultra-dense AI signal display: direction dial, confidence sparkline,
multi-horizon probability matrix, gate attention heatmap, top features,
MoE routing visualization.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.sparkline import render_confidence_sparkline
from aphelion.tui.widgets.gauge import render_gauge, render_mini_bar
from aphelion.tui.widgets.heatmap import render_feature_heatmap, render_gate_weights

_DIR_STYLE = {
    "LONG": "bold bright_green",
    "SHORT": "bold bright_red",
    "FLAT": "dim bright_yellow",
}
_DIR_ICON = {
    "LONG": "▲",
    "SHORT": "▼",
    "FLAT": "◆",
}

_SUB_MODEL_NAMES = ["TFT", "LSTM", "CNN", "MoE"]
_MOE_EXPERT_NAMES = ["TREND", "RANGE", "VOL_EXP", "NEWS"]


def _prob_matrix(probs_5m: list[float], probs_15m: list[float], probs_1h: list[float]) -> Table:
    """Render a probability matrix table like a Bloomberg data grid."""
    t = Table(
        show_header=True,
        header_style="bold bright_cyan",
        expand=True,
        padding=(0, 1),
        show_edge=False,
    )
    t.add_column("TF", style="bright_white", width=5)
    t.add_column("SHORT", justify="center", width=8)
    t.add_column("FLAT", justify="center", width=8)
    t.add_column("LONG", justify="center", width=8)
    t.add_column("Visual", width=16)

    for label, probs in [("5m", probs_5m), ("15m", probs_15m), ("1h", probs_1h)]:
        short_pct = probs[0] * 100
        flat_pct = probs[1] * 100
        long_pct = probs[2] * 100

        # Color based on dominant direction
        s_style = "bold bright_red" if short_pct > 50 else "red"
        f_style = "bold bright_yellow" if flat_pct > 50 else "yellow"
        l_style = "bold bright_green" if long_pct > 50 else "green"

        # Visual bar: weighted bar showing direction
        vis = Text()
        sw = int(probs[0] * 14)
        fw = int(probs[1] * 14)
        lw = 14 - sw - fw
        vis.append("█" * sw, style="red")
        vis.append("█" * fw, style="yellow")
        vis.append("█" * lw, style="green")

        t.add_row(
            Text(label, style="bold"),
            Text(f"{short_pct:.1f}%", style=s_style),
            Text(f"{flat_pct:.1f}%", style=f_style),
            Text(f"{long_pct:.1f}%", style=l_style),
            vis,
        )
    return t


def build_hydra_panel(state: TUIState) -> Panel:
    """Build the HYDRA signal display panel (Bloomberg-grade)."""
    h = state.hydra

    # ── Top section: Direction + Confidence ──
    top = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    top.add_column("signal", ratio=3)
    top.add_column("confidence", ratio=4)

    dir_style = _DIR_STYLE.get(h.direction, "white")
    icon = _DIR_ICON.get(h.direction, "?")

    signal = Text()
    signal.append(f" {icon} ", style=dir_style)
    signal.append(f"{h.direction}", style=dir_style)
    signal.append(f"  Conf:", style="bright_white")
    conf_style = "bright_green" if h.confidence >= 0.7 else (
        "bright_yellow" if h.confidence >= 0.4 else "bright_red"
    )
    signal.append(f" {h.confidence:.1%}", style=f"bold {conf_style}")
    signal.append(f"  Unc:", style="dim")
    signal.append(f" {h.uncertainty:.3f}", style="dim")
    if h.signal_count > 0:
        signal.append(f"  #{h.signal_count}", style="dim")

    # Confidence sparkline
    conf_spark = Text("  ")
    conf_spark.append_text(render_confidence_sparkline(list(h.confidence_history), width=30))

    top.add_row(signal, conf_spark)

    # ── Horizon agreement ──
    agree = Text()
    agree.append("  Agreement: ", style="bright_white")
    agree.append_text(render_gauge(
        h.horizon_agreement, 1.0, width=18, warn_pct=0.6, crit_pct=0.8, invert=True
    ))

    # ── Probability matrix ──
    prob_table = _prob_matrix(h.probs_5m, h.probs_15m, h.probs_1h)

    # ── Gate + MoE ──
    bottom = Table(show_header=False, expand=True, show_edge=False, padding=(0, 0))
    bottom.add_column("gates", ratio=1)
    bottom.add_column("moe", ratio=1)

    gate_table = render_gate_weights(h.gate_weights, _SUB_MODEL_NAMES)
    moe_table = render_gate_weights(h.moe_routing, _MOE_EXPERT_NAMES)

    bottom.add_row(
        Panel(gate_table, title="[cyan]Gate Attention[/]", border_style="dim cyan", padding=(0, 0)),
        Panel(moe_table, title="[magenta]MoE Routing[/]", border_style="dim magenta", padding=(0, 0)),
    )

    # ── Top features heatmap ──
    feat_panel = Panel(
        render_feature_heatmap(h.top_features, max_features=6),
        title="[bright_white]Feature Importance[/]",
        border_style="dim white",
        padding=(0, 0),
    ) if h.top_features else Text("  No feature data", style="dim")

    content = Group(top, agree, prob_table, bottom, feat_panel)

    return Panel(
        content,
        title="[bold bright_yellow]⚡ HYDRA AI Signal[/]",
        border_style="bright_yellow",
    )
