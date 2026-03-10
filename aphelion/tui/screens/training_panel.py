"""
APHELION TUI — Training Screen (F9)  [Phase 23]

Train HYDRA directly from the TUI.  Presets, per-model progress table,
loss curve sparkline, background training via TrainingController.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.controller import TrainingProgress, TrainingState
from aphelion.tui.widgets.progress_bar import (
    render_epoch_progress,
    render_loss_sparkline,
)


TRAINING_PRESETS = [
    {
        "key": "1",
        "name": "Quick test",
        "description": "500 bars, 2 epochs, ~2min",
        "epochs": 2,
        "data_source": "synthetic",
        "bars": 500,
    },
    {
        "key": "2",
        "name": "Full synthetic",
        "description": "10k bars, 20 epochs, ~15min",
        "epochs": 20,
        "data_source": "synthetic",
        "bars": 10_000,
    },
    {
        "key": "3",
        "name": "Full real data",
        "description": "2.6M bars, 100 epochs, ~75min ← RECOMMENDED",
        "epochs": 100,
        "data_source": "real",
        "bars": 2_600_000,
    },
]


def build_training_panel(
    progress: TrainingProgress | None = None,
    data_source: str = "synthetic",
    data_path: str = "",
    detected_bars: int = 0,
    gpu_info: str = "No GPU detected",
    selected_preset: int = -1,
) -> Panel:
    """Build the F9 Training screen.

    Parameters
    ----------
    progress : TrainingProgress, optional
        Current training status. None if idle.
    data_source : str
        "synthetic" or "real".
    data_path : str
        Path to real data file.
    detected_bars : int
        Number of bars detected in data file.
    gpu_info : str
        GPU description string.
    selected_preset : int
        Index of the selected preset (-1 = none).
    """
    progress = progress or TrainingProgress()
    parts: list = []

    # ── Data Source ──
    parts.append(Text("  DATA SOURCE", style="bold bright_yellow"))
    src_table = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    src_table.add_column("radio", width=4)
    src_table.add_column("label", ratio=2)
    src_table.add_column("detail", ratio=3)

    syn_style = "bright_white" if data_source == "synthetic" else "dim"
    real_style = "bright_white" if data_source == "real" else "dim"

    src_table.add_row(
        Text("  ●" if data_source == "synthetic" else "  ○", style=syn_style),
        Text("Synthetic data", style=syn_style),
        Text("(no MT5 needed, quick validation)", style="dim"),
    )
    src_table.add_row(
        Text("  ●" if data_source == "real" else "  ○", style=real_style),
        Text("Real data", style=real_style),
        Text(f"Path: {data_path or 'Not set'}" + (f"  ({detected_bars:,} bars)" if detected_bars else ""), style="dim"),
    )
    parts.append(src_table)
    parts.append(Text(""))

    # ── Quick Presets ──
    parts.append(Text("  QUICK PRESETS:", style="bold bright_yellow"))
    for i, preset in enumerate(TRAINING_PRESETS):
        selected = (i == selected_preset)
        style = "bold bright_white" if selected else "bright_white"
        marker = "▶" if selected else " "
        recommended = " ← RECOMMENDED" if "RECOMMENDED" in preset["description"] else ""
        parts.append(Text(
            f"  {marker} [{preset['key']}] {preset['name']} ({preset['description']}){recommended}",
            style=style,
        ))
    parts.append(Text(""))

    # ── Training Config ──
    cfg_table = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    cfg_table.add_column("k", ratio=1)
    cfg_table.add_column("v", ratio=2)
    cfg_table.add_row(
        Text("  Epochs", style="bright_white"),
        Text(str(progress.total_epochs), style="bright_cyan"),
    )
    cfg_table.add_row(
        Text("  GPU", style="bright_white"),
        Text(gpu_info, style="bright_cyan"),
    )
    parts.append(cfg_table)
    parts.append(Text(""))

    # ── Training Progress ──
    if progress.state != TrainingState.IDLE:
        parts.append(Text("  TRAINING PROGRESS", style="bold bright_yellow"))

        # Epoch bar
        bar = render_epoch_progress(
            current=progress.current_epoch,
            total=progress.total_epochs,
            elapsed=progress.elapsed_seconds,
        )
        parts.append(bar)
        parts.append(Text(""))

        # Per-model table
        if progress.model_metrics:
            model_table = Table(
                show_header=True, expand=True, show_edge=True,
                padding=(0, 1), border_style="dim",
            )
            model_table.add_column("MODEL", style="bright_white", ratio=2)
            model_table.add_column("TRAIN LOSS", justify="right", style="bright_cyan", ratio=1)
            model_table.add_column("VAL LOSS", justify="right", style="bright_cyan", ratio=1)
            model_table.add_column("SHARPE", justify="right", style="bright_cyan", ratio=1)
            model_table.add_column("STATUS", ratio=1)

            for name, metrics in progress.model_metrics.items():
                status_text = Text("✅ Training", style="bright_green")
                model_table.add_row(
                    name,
                    f"{metrics.get('train_loss', 0):.3f}",
                    f"{metrics.get('val_loss', 0):.3f}",
                    f"{metrics.get('sharpe', 0):.2f}",
                    status_text,
                )
            parts.append(model_table)
        parts.append(Text(""))

        # Loss sparkline
        if progress.loss_history:
            parts.append(render_loss_sparkline(progress.loss_history))
        parts.append(Text(""))

        # Status message
        if progress.message:
            style = "bright_green" if progress.state == TrainingState.COMPLETE else "bright_yellow"
            parts.append(Text(f"  {progress.message}", style=style))

    # ── Footer ──
    footer = Text()
    if progress.state == TrainingState.IDLE:
        footer.append("\n  [ENTER]", style="bold bright_cyan")
        footer.append(" Start Training   ", style="dim")
    elif progress.state == TrainingState.RUNNING:
        footer.append("\n  [P]", style="bold bright_cyan")
        footer.append(" Pause   ", style="dim")
        footer.append("[CTRL+C]", style="bold bright_cyan")
        footer.append(" Abort   ", style="dim")
    elif progress.state == TrainingState.PAUSED:
        footer.append("\n  [ENTER]", style="bold bright_cyan")
        footer.append(" Resume   ", style="dim")
        footer.append("[CTRL+C]", style="bold bright_cyan")
        footer.append(" Abort   ", style="dim")

    footer.append("[L]", style="bold bright_cyan")
    footer.append(" Load checkpoint   ", style="dim")
    footer.append("[ESC]", style="bold bright_cyan")
    footer.append(" Back", style="dim")
    parts.append(footer)

    content = Group(*parts)

    return Panel(
        content,
        title="[bold bright_white]HYDRA TRAINING CENTER[/]  [dim][F9][/]",
        border_style="bright_magenta",
        expand=True,
    )
