"""
APHELION TUI — Evolution Panel  (v2 — Bloomberg-grade)

Displays PROMETHEUS genome evolution, FORGE optimization,
CIPHER feature importance, and ZEUS stress test status.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.gauge import render_gauge


def build_evolution_panel(state: TUIState) -> Panel:
    """Build the evolution/optimization panel."""

    # Evolution data from extended state
    generation = getattr(state, "evo_generation", 0)
    best_fitness = getattr(state, "evo_best_fitness", 0.0)
    species_count = getattr(state, "evo_species_count", 0)
    forge_last_run = getattr(state, "forge_last_run", "Never")
    forge_status = getattr(state, "forge_status", "IDLE")
    cipher_top_features = getattr(state, "cipher_top_features", [])
    zeus_last_test = getattr(state, "zeus_last_test", "Never")
    zeus_result = getattr(state, "zeus_result", "N/A")
    auto_opt_runs = getattr(state, "auto_opt_runs", 0)
    auto_opt_applied = getattr(state, "auto_opt_applied", 0)

    # ── Status banner ──
    banner = Text()
    if forge_status == "RUNNING":
        banner.append("  ⟳ OPTIMIZATION IN PROGRESS  ", style="bold bright_yellow on rgb(60,40,0)")
    else:
        banner.append("  ● EVOLUTION IDLE  ", style="bold bright_green on rgb(0,40,0)")

    # ── PROMETHEUS stats ──
    prom = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    prom.add_column("k", ratio=1)
    prom.add_column("v", ratio=2)

    prom.add_row(Text("  Generation", style="bright_white"), Text(str(generation), style="bright_cyan"))
    prom.add_row(Text("  Best Fitness", style="bright_white"), Text(f"{best_fitness:.4f}", style="bright_green"))
    prom.add_row(Text("  Species", style="bright_white"), Text(str(species_count), style="bright_cyan"))

    # ── FORGE stats ──
    forge = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    forge.add_column("k", ratio=1)
    forge.add_column("v", ratio=2)

    forge.add_row(Text("  Last FORGE", style="bright_white"), Text(str(forge_last_run), style="dim"))
    forge.add_row(Text("  AutoOpt Runs", style="bright_white"), Text(str(auto_opt_runs), style="bright_cyan"))
    forge.add_row(Text("  Applied", style="bright_white"), Text(str(auto_opt_applied), style="bright_green"))

    # ── CIPHER top features ──
    feat_table = Table(show_header=True, expand=True, show_edge=False, padding=(0, 1))
    feat_table.add_column("#", width=3, style="dim")
    feat_table.add_column("Feature", style="bright_white")
    feat_table.add_column("Imp", justify="right", style="bright_cyan")

    for i, (fname, imp) in enumerate(cipher_top_features[:6], 1):
        feat_table.add_row(str(i), fname, f"{imp:.3f}")

    if not cipher_top_features:
        feat_table.add_row("—", "No data yet", "—")

    # ── ZEUS ──
    zeus_line = Text()
    zeus_line.append(f"  ZEUS: {zeus_result}", style="bright_white")
    zeus_line.append(f"  (last: {zeus_last_test})", style="dim")

    content = Group(banner, Text(""), prom, Text(""), forge, Text(""), feat_table, Text(""), zeus_line)
    return Panel(
        content,
        title="[bold bright_white]EVOLUTION & OPTIMIZATION[/]",
        border_style="bright_green",
        expand=True,
    )
