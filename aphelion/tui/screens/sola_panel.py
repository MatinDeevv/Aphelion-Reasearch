"""
APHELION TUI — SOLA Governance Panel  (v2 — Bloomberg-grade)

Displays SOLA mode, edge confidence, veto activity, module rankings,
and self-improvement cycle status.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.gauge import render_gauge


def build_sola_panel(state: TUIState) -> Panel:
    """Build the SOLA governance panel."""

    # We read SOLA data from state — extend TUIState if needed
    # For now, use state-level attributes with safe getattr defaults
    mode = getattr(state, "sola_mode", "ACTIVE")
    edge_conf = getattr(state, "sola_edge_confidence", 1.0)
    veto_count = getattr(state, "sola_veto_count", 0)
    improvement_cycle = getattr(state, "sola_improvement_cycle", 0)
    module_rankings = getattr(state, "sola_module_rankings", [])
    black_swan = getattr(state, "sola_black_swan", False)

    # ── Mode banner ──
    mode_styles = {
        "ACTIVE": ("bright_green", "rgb(0,40,0)"),
        "CAUTIOUS": ("bright_yellow", "rgb(60,40,0)"),
        "DEFENSIVE": ("bright_red", "rgb(60,0,0)"),
        "LOCKDOWN": ("bold bright_red", "rgb(80,0,0)"),
    }
    fg, bg = mode_styles.get(mode, ("bright_white", "rgb(30,30,30)"))

    banner = Text()
    banner.append(f"  ● SOLA MODE: {mode}  ", style=f"{fg} on {bg}")
    if black_swan:
        banner.append("  ⚠ BLACK SWAN  ", style="bold bright_red blink")

    # ── Edge confidence gauge ──
    gauges = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    gauges.add_column("label", ratio=1)
    gauges.add_column("gauge", ratio=3)

    gauges.add_row(
        Text("  Edge", style="bright_white"),
        render_gauge(edge_conf * 100, 100.0, width=22, label=f"{edge_conf:.0%}"),
    )

    # ── Stats ──
    stats = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    stats.add_column("k", ratio=1)
    stats.add_column("v", ratio=2)

    stats.add_row(Text("  Vetoes", style="bright_white"), Text(str(veto_count), style="bright_yellow"))
    stats.add_row(Text("  Improvement", style="bright_white"), Text(f"Cycle #{improvement_cycle}", style="bright_cyan"))

    # ── Module rankings ──
    rank_table = Table(show_header=True, expand=True, show_edge=False, padding=(0, 1))
    rank_table.add_column("#", width=3, style="dim")
    rank_table.add_column("Module", style="bright_white")
    rank_table.add_column("Score", justify="right", style="bright_cyan")

    for i, (name, score) in enumerate(module_rankings[:8], 1):
        score_style = "bright_green" if score > 0 else ("bright_red" if score < 0 else "dim")
        rank_table.add_row(str(i), name, Text(f"{score:.3f}", style=score_style))

    content = Group(banner, Text(""), gauges, Text(""), stats, Text(""), rank_table)
    return Panel(
        content,
        title="[bold bright_white]SOLA GOVERNANCE[/]",
        border_style="bright_magenta",
        expand=True,
    )
