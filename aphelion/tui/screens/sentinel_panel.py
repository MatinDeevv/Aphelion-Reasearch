"""
APHELION TUI — SENTINEL Risk Panel  (v2 — Bloomberg-grade)

Multi-gauge risk dashboard: circuit breaker indicators, drawdown gauge,
exposure meter, position utilisation, risk heat status.
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.state import TUIState
from aphelion.tui.widgets.gauge import render_gauge, render_breaker_indicator
from aphelion.tui.widgets.sparkline import render_sparkline


def build_sentinel_panel(state: TUIState) -> Panel:
    """Build the SENTINEL risk status panel (Bloomberg-grade)."""
    s = state.sentinel

    # ── Status banner ──
    status = Text()
    if s.trading_allowed and not s.circuit_breaker_active:
        status.append("  ● TRADING ACTIVE  ", style="bold bright_green on rgb(0,40,0)")
    elif s.circuit_breaker_active:
        status.append("  ⊗ CIRCUIT BREAKER ACTIVE  ", style="bold bright_red on rgb(60,0,0)")
    else:
        status.append("  ○ TRADING SUSPENDED  ", style="bold bright_yellow on rgb(60,40,0)")

    # ── Circuit breaker indicators ──
    breakers = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    breakers.add_column()
    breakers.add_row(render_breaker_indicator("L1 Cooldown", s.l1_triggered))
    breakers.add_row(render_breaker_indicator("L2 Loss Limit", s.l2_triggered))
    breakers.add_row(render_breaker_indicator("L3 Emergency", s.l3_triggered))

    # ── Risk gauges ──
    gauges = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    gauges.add_column("label", ratio=1)
    gauges.add_column("gauge", ratio=3)

    # Drawdown gauge
    dd_pct = s.daily_drawdown_pct * 100
    max_dd = 5.0  # 5% max daily DD threshold
    gauges.add_row(
        Text("  DD%", style="bright_white"),
        render_gauge(dd_pct, max_dd, width=22, label="", warn_pct=0.5, crit_pct=0.8),
    )

    # Exposure gauge
    exp_pct = s.total_exposure_pct * 100
    max_exp = s.max_exposure_pct * 100
    gauges.add_row(
        Text("  EXP%", style="bright_white"),
        render_gauge(exp_pct, max_exp, width=22, label="", warn_pct=0.6, crit_pct=0.85),
    )

    # Position utilisation gauge
    gauges.add_row(
        Text("  POS", style="bright_white"),
        render_gauge(
            float(s.open_positions), float(s.max_positions),
            width=22, label="", warn_pct=0.6, crit_pct=0.9,
        ),
    )

    # ── Drawdown sparkline (if history available) ──
    dd_hist = list(s.drawdown_history)
    dd_spark_text = Text("  DD History: ", style="dim")
    if dd_hist:
        dd_spark_text.append_text(render_sparkline(
            dd_hist, width=30,
            color_positive="bright_red",
            color_negative="bright_green",
            color_neutral="bright_yellow",
        ))
    else:
        dd_spark_text.append("─" * 30, style="dim")

    # ── Summary stats ──
    stats = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    stats.add_column(ratio=1)
    stats.add_column(ratio=1)

    left = Text()
    left.append(f"  Peak: ", style="dim")
    left.append(f"${s.session_peak_equity:,.2f}", style="bright_white")

    right = Text()
    if s.breaker_since:
        elapsed = ""
        right.append(f"  Breaker since: ", style="dim")
        right.append(s.breaker_since.strftime("%H:%M:%S"), style="bright_red")
    else:
        right.append("  No breakers active", style="dim green")

    stats.add_row(left, right)

    content = Group(status, breakers, gauges, dd_spark_text, stats)

    return Panel(
        content,
        title="[bold bright_red]◈ SENTINEL Risk Control[/]",
        border_style="bright_red",
    )
