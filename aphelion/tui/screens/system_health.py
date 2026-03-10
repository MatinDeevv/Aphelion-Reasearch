"""
APHELION TUI — System Health Panel  (v2 — Bloomberg-grade)

Displays CPU, memory, latency, feed status, and module health.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.gauge import render_gauge


def build_system_health_panel(state: TUIState) -> Panel:
    """Build the system health panel."""

    # ── Feed status banner ──
    status = Text()
    if state.feed_connected:
        status.append("  ● FEED CONNECTED  ", style="bold bright_green on rgb(0,40,0)")
    else:
        status.append("  ○ FEED DISCONNECTED  ", style="bold bright_red on rgb(60,0,0)")

    status.append(f"  mode={state.feed_mode}", style="dim")
    status.append(f"  ticks/min={state.feed_ticks_per_min:.0f}", style="dim")
    if state.feed_reconnect_count > 0:
        status.append(f"  reconnects={state.feed_reconnect_count}", style="bright_yellow")

    # ── System metrics ──
    metrics = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    metrics.add_column("label", ratio=1)
    metrics.add_column("gauge", ratio=3)

    # CPU
    cpu_style = "bright_green" if state.cpu_usage < 60 else ("bright_yellow" if state.cpu_usage < 85 else "bright_red")
    metrics.add_row(
        Text("  CPU", style="bright_white"),
        render_gauge(state.cpu_usage, 100.0, width=22, label=f"{state.cpu_usage:.0f}%"),
    )

    # Memory
    metrics.add_row(
        Text("  MEM", style="bright_white"),
        render_gauge(state.memory_mb, 1024.0, width=22, label=f"{state.memory_mb:.0f}MB"),
    )

    # Latency
    lat_style = "bright_green" if state.latency_ms < 50 else ("bright_yellow" if state.latency_ms < 200 else "bright_red")
    metrics.add_row(
        Text("  LAT", style="bright_white"),
        render_gauge(state.latency_ms, 500.0, width=22, label=f"{state.latency_ms:.0f}ms"),
    )

    # ── Uptime / bars ──
    info = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    info.add_column("k", ratio=1)
    info.add_column("v", ratio=2)

    hrs = state.uptime_seconds / 3600
    info.add_row(Text("  Uptime", style="bright_white"), Text(f"{hrs:.1f}h", style="bright_cyan"))
    info.add_row(Text("  Bars", style="bright_white"), Text(str(state.bars_processed), style="bright_cyan"))
    info.add_row(Text("  Session", style="bright_white"), Text(state.current_session, style="bright_cyan"))
    info.add_row(
        Text("  Sentinel Rej", style="bright_white"),
        Text(str(state.sentinel_rejections), style="bright_yellow" if state.sentinel_rejections > 0 else "dim"),
    )

    content = Group(status, Text(""), metrics, Text(""), info)
    return Panel(
        content,
        title="[bold bright_white]SYSTEM HEALTH[/]",
        border_style="bright_blue",
        expand=True,
    )
