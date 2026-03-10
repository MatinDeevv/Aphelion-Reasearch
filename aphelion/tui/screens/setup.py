"""
APHELION TUI — Setup / Configuration Screen (F7)  [Phase 23]

Interactive form for all APHELION configuration parameters.
Tab between fields. Press [S] to save. Press [R] to reset to defaults.
"""

from __future__ import annotations

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aphelion.tui.config import AphelionConfig
from aphelion.tui.widgets.form_field import FormField, FormState, render_form


def build_setup_form(config: AphelionConfig) -> FormState:
    """Create a FormState pre-populated from an AphelionConfig."""
    fields = [
        # Trading section
        FormField(
            key="trading.symbol", label="Symbol",
            value=config.trading.symbol, section="Trading",
        ),
        FormField(
            key="trading.mode", label="Mode",
            value=config.trading.mode,
            field_type="select", options=["paper", "simulated", "backtest"],
            section="Trading",
        ),
        FormField(
            key="trading.capital", label="Starting Capital",
            value=f"{config.trading.capital:.2f}",
            field_type="number", section="Trading",
        ),
        FormField(
            key="trading.warmup_bars", label="Warmup Bars",
            value=str(config.trading.warmup_bars),
            field_type="number", section="Trading",
        ),

        # MT5 section
        FormField(
            key="mt5.login", label="Login",
            value=str(config.mt5.login) if config.mt5.login else "",
            field_type="number", section="MT5 Connection",
        ),
        FormField(
            key="mt5.password", label="Password",
            value=config.mt5.password,
            field_type="password", section="MT5 Connection",
        ),
        FormField(
            key="mt5.server", label="Server",
            value=config.mt5.server, section="MT5 Connection",
        ),
        FormField(
            key="mt5.terminal_path", label="Terminal Path",
            value=config.mt5.terminal_path, section="MT5 Connection",
        ),

        # Risk section
        FormField(
            key="risk.max_daily_dd", label="Max Daily DD",
            value=f"{config.risk.max_daily_dd * 100:.1f}%",
            field_type="text", section="Risk",
        ),
        FormField(
            key="risk.max_exposure", label="Max Exposure",
            value=f"{config.risk.max_exposure * 100:.1f}%",
            field_type="text", section="Risk",
        ),
        FormField(
            key="risk.max_positions", label="Max Positions",
            value=str(config.risk.max_positions),
            field_type="number", section="Risk",
        ),
        FormField(
            key="risk.risk_per_trade", label="Risk Per Trade",
            value=f"{config.risk.risk_per_trade * 100:.1f}%",
            field_type="text", section="Risk",
        ),

        # HYDRA section
        FormField(
            key="hydra.checkpoint", label="Checkpoint",
            value=config.hydra.checkpoint, section="HYDRA",
        ),
        FormField(
            key="hydra.min_confidence", label="Min Confidence",
            value=f"{config.hydra.min_confidence:.2f}",
            field_type="number", section="HYDRA",
        ),
    ]

    return FormState(fields=fields)


def apply_form_to_config(form: FormState, config: AphelionConfig) -> list[str]:
    """Apply form field values back to config. Returns validation errors."""
    errors = form.validate_all()
    if errors:
        return errors

    d = form.to_dict()

    config.trading.symbol = d.get("trading.symbol", config.trading.symbol)
    config.trading.mode = d.get("trading.mode", config.trading.mode)

    try:
        config.trading.capital = float(d.get("trading.capital", str(config.trading.capital)))
    except ValueError:
        errors.append("Invalid capital value")

    try:
        config.trading.warmup_bars = int(float(d.get("trading.warmup_bars", str(config.trading.warmup_bars))))
    except ValueError:
        errors.append("Invalid warmup bars value")

    login_str = d.get("mt5.login", "")
    if login_str:
        try:
            config.mt5.login = int(float(login_str))
        except ValueError:
            errors.append("Invalid MT5 login")

    config.mt5.password = d.get("mt5.password", config.mt5.password)
    config.mt5.server = d.get("mt5.server", config.mt5.server)
    config.mt5.terminal_path = d.get("mt5.terminal_path", config.mt5.terminal_path)

    # Parse percentage fields
    for pct_key, attr, obj in [
        ("risk.max_daily_dd", "max_daily_dd", config.risk),
        ("risk.max_exposure", "max_exposure", config.risk),
        ("risk.risk_per_trade", "risk_per_trade", config.risk),
    ]:
        raw = d.get(pct_key, "")
        if raw:
            try:
                val = float(raw.replace("%", "").strip()) / 100.0
                setattr(obj, attr, val)
            except ValueError:
                errors.append(f"Invalid {pct_key}")

    try:
        config.risk.max_positions = int(float(d.get("risk.max_positions", str(config.risk.max_positions))))
    except ValueError:
        errors.append("Invalid max positions")

    config.hydra.checkpoint = d.get("hydra.checkpoint", config.hydra.checkpoint)

    try:
        config.hydra.min_confidence = float(d.get("hydra.min_confidence", str(config.hydra.min_confidence)))
    except ValueError:
        errors.append("Invalid HYDRA min confidence")

    return errors


def build_setup_panel(
    form: FormState,
    config: AphelionConfig,
    save_message: str = "",
) -> Panel:
    """Build the F7 Setup screen as a Rich Panel.

    Parameters
    ----------
    form : FormState
        Current form state with field values and active index.
    config : AphelionConfig
        The backing config object.
    save_message : str
        Status message shown after save (e.g., "Saved" or error).
    """
    parts: list = []

    # ── Section sidebar + form ──
    layout_table = Table(show_header=False, expand=True, show_edge=False, padding=(0, 1))
    layout_table.add_column("sidebar", ratio=1)
    layout_table.add_column("form", ratio=3)

    # Sidebar
    sidebar = Text()
    sidebar.append("  SECTION\n", style="bold bright_yellow")
    sidebar.append("  " + "─" * 20 + "\n", style="dim")
    for section in form.sections:
        active = form.active_field and form.active_field.section == section
        style = "bold bright_white" if active else "dim"
        prefix = "▶ " if active else "  "
        sidebar.append(f"  {prefix}{section}\n", style=style)

    # Form fields
    form_table = render_form(form, title="Settings")

    layout_table.add_row(sidebar, form_table)
    parts.append(layout_table)

    # ── Status bar ──
    if save_message:
        parts.append(Text(f"\n  {save_message}", style="bright_green" if "Saved" in save_message else "bright_red"))

    # ── Footer ──
    footer = Text()
    footer.append("\n  [TAB]", style="bold bright_cyan")
    footer.append(" Next field   ", style="dim")
    footer.append("[S]", style="bold bright_cyan")
    footer.append(" Save & Apply   ", style="dim")
    footer.append("[R]", style="bold bright_cyan")
    footer.append(" Reset to defaults   ", style="dim")
    footer.append("[ESC]", style="bold bright_cyan")
    footer.append(" Back", style="dim")
    parts.append(footer)

    content = Group(*parts)

    return Panel(
        content,
        title="[bold bright_white]APHELION SETUP[/]  [dim][F7][/]",
        border_style="bright_cyan",
        expand=True,
    )
