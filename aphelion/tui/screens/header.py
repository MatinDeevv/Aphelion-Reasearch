"""
APHELION TUI — Header Bar  (v2 — Bloomberg-grade)

Top bar showing: session name │ live price ticker │ session │ clock │ status
Information-dense single-line header like Bloomberg's top bar.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.ticker import render_price_ticker


def build_header(state: TUIState) -> Panel:
    """Build the top header bar panel."""
    t = Table(show_header=False, expand=True, show_edge=False, padding=(0, 0))
    t.add_column("left", ratio=3)
    t.add_column("center", ratio=5)
    t.add_column("right", ratio=3, justify="right")

    # ── Left: Session identity ──
    left = Text()
    left.append(" ◆ APHELION ", style="bold bright_white on rgb(0,80,180)")
    left.append(f" {state.session_name} ", style="bold bright_cyan")

    # ── Center: Price ticker ──
    center = render_price_ticker(state.price)

    # ── Right: Clock + session + status ──
    right = Text()
    now = state.current_time or datetime.now(timezone.utc)
    right.append(now.strftime("%H:%M:%S UTC"), style="bold rgb(255,176,0)")
    right.append(" │ ", style="dim")
    right.append(f"{state.current_session}", style="bold bright_white")
    right.append(" │ ", style="dim")

    if state.market_open:
        right.append("● LIVE", style="bold bright_green")
    else:
        right.append("○ CLOSED", style="dim red")

    right.append(" │ ", style="dim")
    right.append(f"Bar #{state.bars_processed}", style="dim")

    t.add_row(left, center, right)

    # Second row: system stats + alerts
    t2 = Table(show_header=False, expand=True, show_edge=False, padding=(0, 0))
    t2.add_column("stats", ratio=4)
    t2.add_column("alerts", ratio=3, justify="right")

    stats = Text()
    stats.append("  CPU:", style="dim")
    cpu_color = "bright_green" if state.cpu_usage < 60 else (
        "bright_yellow" if state.cpu_usage < 85 else "bright_red"
    )
    stats.append(f"{state.cpu_usage:.0f}%", style=cpu_color)
    stats.append("  MEM:", style="dim")
    stats.append(f"{state.memory_mb:.0f}MB", style="dim white")
    stats.append("  LAT:", style="dim")
    lat_color = "bright_green" if state.latency_ms < 50 else (
        "bright_yellow" if state.latency_ms < 200 else "bright_red"
    )
    stats.append(f"{state.latency_ms:.0f}ms", style=lat_color)
    if state.uptime_seconds > 0:
        h, rem = divmod(int(state.uptime_seconds), 3600)
        m, s = divmod(rem, 60)
        stats.append(f"  UP:{h:02d}:{m:02d}:{s:02d}", style="dim")

    alert_text = Text()
    unack = state.unacknowledged_alerts
    if unack:
        crit = sum(1 for a in unack if a.severity == "CRITICAL")
        warn = sum(1 for a in unack if a.severity == "WARNING")
        if crit:
            alert_text.append(f"⊗ {crit} CRITICAL ", style="bold bright_red")
        if warn:
            alert_text.append(f"⊘ {warn} WARNING ", style="bold bright_yellow")
    else:
        alert_text.append("✓ No alerts", style="dim green")

    t2.add_row(stats, alert_text)

    return Panel(
        Group(t, t2),
        border_style="rgb(40,40,100)",
        padding=(0, 0),
    )
