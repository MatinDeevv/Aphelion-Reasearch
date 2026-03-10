"""
APHELION TUI — Backtest Screen (F10)  [Phase 23]

Configure and run backtests directly from the TUI.
Equity curve rendering, results table, export options.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.controller import BacktestProgress
from aphelion.tui.widgets.progress_bar import render_progress_bar


def _render_equity_curve(
    equity_points: list[float],
    width: int = 60,
    height: int = 8,
) -> Text:
    """Render a simple ASCII equity curve."""
    if not equity_points or len(equity_points) < 2:
        return Text("  (No equity data yet)", style="dim")

    mn = min(equity_points)
    mx = max(equity_points)
    rng = mx - mn if mx > mn else 1.0

    # Downsample
    step = max(1, len(equity_points) // width)
    samples = equity_points[::step][:width]

    # Build character grid
    lines: list[list[str]] = [[" "] * len(samples) for _ in range(height)]

    for col, val in enumerate(samples):
        row = int((val - mn) / rng * (height - 1))
        row = max(0, min(row, height - 1))
        lines[height - 1 - row][col] = "█"

    txt = Text()
    # Y-axis labels
    for r in range(height):
        y_val = mx - (r / (height - 1)) * rng if height > 1 else mx
        label = f"  ${y_val / 1000:.0f}k" if y_val >= 1000 else f"  ${y_val:.0f}"
        txt.append(f"{label:>7} ┤", style="dim")
        line_str = "".join(lines[r])
        style = "bright_green" if r < height // 2 else "bright_cyan"
        txt.append(line_str, style=style)
        txt.append("\n")

    txt.append("        └" + "─" * len(samples) + "→\n", style="dim")

    return txt


def build_backtest_panel(
    progress: BacktestProgress | None = None,
    config: dict | None = None,
    equity_curve: list[float] | None = None,
) -> Panel:
    """Build the F10 Backtest screen.

    Parameters
    ----------
    progress : BacktestProgress, optional
        Backtest run progress.
    config : dict, optional
        Backtest configuration.
    equity_curve : list[float], optional
        Equity curve points for chart rendering.
    """
    progress = progress or BacktestProgress()
    config = config or {}
    parts: list = []

    # ── Config section ──
    parts.append(Text("  CONFIG", style="bold bright_yellow"))
    cfg_table = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    cfg_table.add_column("k", ratio=1)
    cfg_table.add_column("v", ratio=2)

    cfg_table.add_row(
        Text("  Symbol", style="bright_white"),
        Text(config.get("symbol", "XAUUSD"), style="bright_cyan"),
    )
    cfg_table.add_row(
        Text("  From", style="bright_white"),
        Text(config.get("start_date", "2023-01-01"), style="bright_cyan"),
    )
    cfg_table.add_row(
        Text("  To", style="bright_white"),
        Text(config.get("end_date", "2025-01-01"), style="bright_cyan"),
    )
    cfg_table.add_row(
        Text("  Capital", style="bright_white"),
        Text(f"${config.get('capital', 10_000):,.0f}", style="bright_cyan"),
    )
    cfg_table.add_row(
        Text("  Commission", style="bright_white"),
        Text(f"{config.get('commission_pips', 0.35)} pips", style="bright_cyan"),
    )
    cfg_table.add_row(
        Text("  Data", style="bright_white"),
        Text(config.get("data_path", "(no data selected)"), style="dim"),
    )
    parts.append(cfg_table)
    parts.append(Text(""))

    # ── Action bar ──
    action = Text()
    action.append("  [ENTER]", style="bold bright_cyan")
    action.append(" Run Backtest    ", style="dim")
    action.append("[W]", style="bold bright_cyan")
    action.append(" Walk-Forward    ", style="dim")
    action.append("[M]", style="bold bright_cyan")
    action.append(" Monte Carlo    ", style="dim")
    action.append("[CTRL+C]", style="bold bright_cyan")
    action.append(" Stop", style="dim")
    parts.append(action)
    parts.append(Text(""))

    # ── Progress / Results ──
    if progress.running:
        parts.append(Text("  RUNNING...", style="bold bright_yellow"))
        bar = render_progress_bar(
            label="Backtest",
            percent=progress.bars_processed / max(progress.total_bars, 1),
            message=progress.message,
            elapsed=progress.elapsed_seconds,
        )
        parts.append(bar)
    elif progress.results:
        parts.append(Text("  RESULTS", style="bold bright_yellow"))
        parts.append(Text(""))

        res = progress.results
        results_table = Table(
            show_header=True, expand=True, show_edge=True,
            padding=(0, 1), border_style="dim",
        )
        results_table.add_column("PERFORMANCE", style="bright_white", ratio=1)
        results_table.add_column("RISK", style="bright_white", ratio=1)
        results_table.add_column("TRADE STATS", style="bright_white", ratio=1)

        perf_lines = [
            f"Total Return: {res.get('total_return_pct', 0):+.0f}%",
            f"Sharpe: {res.get('sharpe', 0):.2f}",
            f"Final Equity: ${res.get('final_equity', 0):,.0f}",
        ]
        risk_lines = [
            f"Max Drawdown: {res.get('max_drawdown_pct', 0):.1f}%",
            f"Sortino: {res.get('sortino', 0):.2f}",
            f"Calmar: {res.get('calmar', 0):.2f}",
        ]
        trade_lines = [
            f"Total Trades: {res.get('total_trades', 0):,}",
            f"Win Rate: {res.get('win_rate', 0):.1f}%",
            f"Profit Factor: {res.get('profit_factor', 0):.2f}",
        ]

        results_table.add_row(
            "\n".join(perf_lines),
            "\n".join(risk_lines),
            "\n".join(trade_lines),
        )
        parts.append(results_table)
        parts.append(Text(""))

        # Equity curve
        if equity_curve:
            parts.append(Text("  EQUITY CURVE", style="bold bright_yellow"))
            parts.append(_render_equity_curve(equity_curve))

    elif progress.message:
        parts.append(Text(f"  {progress.message}", style="bright_yellow"))

    # ── Footer ──
    footer = Text()
    footer.append("\n  [E]", style="bold bright_cyan")
    footer.append(" Export CSV   ", style="dim")
    footer.append("[P]", style="bold bright_cyan")
    footer.append(" Print Report   ", style="dim")
    footer.append("[S]", style="bold bright_cyan")
    footer.append(" Save to registry   ", style="dim")
    footer.append("[ESC]", style="bold bright_cyan")
    footer.append(" Back", style="dim")
    parts.append(footer)

    content = Group(*parts)

    return Panel(
        content,
        title="[bold bright_white]BACKTEST ENGINE[/]  [dim][F10][/]",
        border_style="bright_blue",
        expand=True,
    )
