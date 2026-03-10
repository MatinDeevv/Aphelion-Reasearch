"""
APHELION TUI — HEPHAESTUS Forge Screen (F8)  [Phase 23 — Wired]

Two input modes:
  1. Paste into the text area below and press [F]
  2. Write code to ``config/hephaestus_input.txt`` and press [F]

Live progress bar.  Deployed strategies table.
"""

from __future__ import annotations

import os

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.controller import ForgeProgress
from aphelion.tui.widgets.progress_bar import render_progress_bar
from aphelion.tui.widgets.text_area import TextAreaState, render_text_area

HEPHAESTUS_INPUT_FILE = os.path.join("config", "hephaestus_input.txt")


def build_hephaestus_panel(
    text_area: TextAreaState,
    forge_progress: ForgeProgress | None = None,
    deployed: list[dict] | None = None,
    rejections: list[dict] | None = None,
    selected_row: int = 0,
) -> Panel:
    """Build the F8 HEPHAESTUS Forge screen.

    Parameters
    ----------
    text_area : TextAreaState
        Editable text area state holding the user's pasted code.
    forge_progress : ForgeProgress, optional
        Current forge operation progress (None if idle).
    deployed : list of dicts, optional
        List of deployed strategies with keys: name, sharpe, wr, mode, trades, status.
    rejections : list of dicts, optional
        List of recently rejected strategies with keys: name, reason.
    selected_row : int
        Currently selected row in the deployed table.
    """
    parts: list = []

    # ── Header ──
    header = Text()
    header.append("  PASTE INDICATOR CODE", style="bold bright_yellow")
    header.append("  (Pine Script / Python / Plain English)\n", style="dim")
    # File input hint
    file_exists = os.path.isfile(HEPHAESTUS_INPUT_FILE)
    file_hint = Text()
    if file_exists:
        file_hint.append("  ✓ File detected: ", style="bright_green")
        file_hint.append(HEPHAESTUS_INPUT_FILE, style="dim")
        file_hint.append("  (will be used when [F] pressed)\n", style="dim")
    else:
        file_hint.append("  ○ Or write code to: ", style="dim")
        file_hint.append(HEPHAESTUS_INPUT_FILE, style="dim")
        file_hint.append("\n", style="dim")
    parts.append(header)
    parts.append(file_hint)

    # ── Text area ──
    ta_panel = render_text_area(text_area, title="Code Input", height=10)
    parts.append(ta_panel)

    # ── Text area shortcuts ──
    ta_footer = Text()
    ta_footer.append("  [F]", style="bold bright_cyan")
    ta_footer.append(" Forge Strategy   ", style="dim")
    ta_footer.append("[CTRL+V]", style="bold bright_cyan")
    ta_footer.append(" Paste   ", style="dim")
    ta_footer.append("[CTRL+X]", style="bold bright_cyan")
    ta_footer.append(" Clear", style="dim")
    parts.append(ta_footer)
    parts.append(Text(""))

    # ── Forge Progress ──
    if forge_progress and not forge_progress.complete:
        parts.append(Text("  FORGE PROGRESS", style="bold bright_yellow"))
        bar = render_progress_bar(
            label=f"▶ {forge_progress.stage}",
            percent=forge_progress.percent,
            message=forge_progress.message,
            elapsed=forge_progress.elapsed_seconds,
        )
        parts.append(bar)
        parts.append(Text(""))
    elif forge_progress and forge_progress.complete:
        if forge_progress.success:
            parts.append(Text(f"  ✅ {forge_progress.message}", style="bold bright_green"))
        else:
            parts.append(Text(f"  ✗ {forge_progress.message}", style="bold bright_red"))
        parts.append(Text(""))

    # ── Deployed Strategies Table ──
    deployed = deployed or []
    dep_header = Text()
    dep_header.append(f"  DEPLOYED STRATEGIES ({len(deployed)} active)\n", style="bold bright_yellow")
    parts.append(dep_header)

    dep_table = Table(
        show_header=True,
        expand=True,
        show_edge=True,
        padding=(0, 1),
        border_style="dim",
    )
    dep_table.add_column("NAME", style="bright_white", ratio=3)
    dep_table.add_column("SHARPE", justify="right", style="bright_cyan", ratio=1)
    dep_table.add_column("WR", justify="right", style="bright_cyan", ratio=1)
    dep_table.add_column("MODE", style="bright_white", ratio=1)
    dep_table.add_column("TRADES", justify="right", style="bright_white", ratio=1)
    dep_table.add_column("STATUS", ratio=1)

    for i, d in enumerate(deployed):
        name = d.get("name", "")
        sharpe = d.get("sharpe", 0)
        wr = d.get("wr", 0)
        mode = d.get("mode", "SHADOW")
        trades = d.get("trades", 0)
        status = d.get("status", "SHADOW")

        status_text = Text()
        if status == "LIVE":
            status_text.append("✅ LIVE", style="bright_green")
        elif status == "SHADOW":
            status_text.append("⏳ SHADOW", style="bright_yellow")
        else:
            status_text.append(status, style="dim")

        row_style = "bold" if i == selected_row else ""
        dep_table.add_row(
            Text(name, style=row_style),
            f"{sharpe:.2f}",
            f"{wr:.0f}%",
            mode,
            str(trades),
            status_text,
        )

    if not deployed:
        dep_table.add_row("No deployed strategies", "", "", "", "", Text("—", style="dim"))

    parts.append(dep_table)

    # ── Recent Rejections ──
    rejections = rejections or []
    if rejections:
        parts.append(Text("\n  RECENT REJECTIONS", style="bold bright_yellow"))
        for r in rejections[:5]:
            name = r.get("name", "")
            reason = r.get("reason", "")
            parts.append(Text(f"  ✗ {name}   {reason}", style="bright_red"))
        parts.append(Text(""))

    # ── Bottom shortcuts ──
    footer = Text()
    footer.append("\n  [D]", style="bold bright_cyan")
    footer.append(" Deploy Selected   ", style="dim")
    footer.append("[R]", style="bold bright_cyan")
    footer.append(" Remove Selected   ", style="dim")
    footer.append("[V]", style="bold bright_cyan")
    footer.append(" View Report   ", style="dim")
    footer.append("[ESC]", style="bold bright_cyan")
    footer.append(" Back", style="dim")
    parts.append(footer)

    content = Group(*parts)

    return Panel(
        content,
        title="[bold bright_white]HEPHAESTUS — Strategy Forge[/]  [dim][F8][/]",
        border_style="rgb(255,100,0)",
        expand=True,
    )
