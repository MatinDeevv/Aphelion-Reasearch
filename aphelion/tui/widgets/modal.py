"""
APHELION TUI — Modal Dialog Widget (Phase 23)

Rich-panel-based modal dialogs for confirmations, alerts, and wizards.
These render as overlaid panels in the dashboard view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rich.align import Align
from rich.panel import Panel
from rich.text import Text


@dataclass
class ModalButton:
    """A button option in a modal dialog."""
    key: str        # keyboard key (e.g. "Y", "N", "ESC")
    label: str      # display text
    action: str     # action identifier for the handler


@dataclass
class ModalState:
    """State for a modal dialog overlay."""
    visible: bool = False
    title: str = ""
    body_lines: list[str] = field(default_factory=list)
    buttons: list[ModalButton] = field(default_factory=list)
    style: str = "bright_white"  # title style
    border_style: str = "bright_yellow"
    result: Optional[str] = None  # set when user picks an option

    def show(
        self,
        title: str,
        body: list[str],
        buttons: list[ModalButton],
        style: str = "bright_white",
        border_style: str = "bright_yellow",
    ) -> None:
        """Display the modal."""
        self.visible = True
        self.title = title
        self.body_lines = body
        self.buttons = buttons
        self.style = style
        self.border_style = border_style
        self.result = None

    def dismiss(self, action: str | None = None) -> None:
        """Dismiss the modal."""
        self.visible = False
        self.result = action


def render_modal(state: ModalState) -> Panel:
    """Render a modal dialog as a Rich Panel."""
    text = Text()
    text.append(f"  {state.title}\n", style=f"bold {state.style}")
    text.append("\n")

    for line in state.body_lines:
        text.append(f"  {line}\n", style="bright_white")

    text.append("\n")
    for btn in state.buttons:
        text.append(f"  [{btn.key}] {btn.label}\n", style="bright_cyan")

    return Panel(
        Align.center(text),
        border_style=state.border_style,
        expand=False,
        width=50,
    )


# ─── Pre-built modals ───────────────────────────────────────────────────────


def build_quit_confirmation_modal(
    open_positions: int = 0,
    unrealized_pnl: float = 0.0,
) -> ModalState:
    """Build the quit confirmation modal shown during an active session."""
    modal = ModalState()
    body = [
        f"Open positions: {open_positions}",
        f"Unrealized PnL: ${unrealized_pnl:+,.2f}",
    ]
    buttons = [
        ModalButton("Y", "Stop & Close All", "stop_close"),
        ModalButton("K", "Stop & Keep Positions", "stop_keep"),
        ModalButton("ESC", "Cancel (keep trading)", "cancel"),
    ]
    modal.show(
        title="STOP TRADING SESSION?",
        body=body,
        buttons=buttons,
        border_style="bright_red",
    )
    return modal


def build_forge_success_modal(
    name: str,
    sharpe: float = 0.0,
    win_rate: float = 0.0,
    trades: int = 0,
) -> ModalState:
    """Build the forge success modal."""
    modal = ModalState()
    body = [
        f"Name:     {name}",
        f"Sharpe:   {sharpe:.2f}",
        f"Win Rate: {win_rate:.0f}%",
        f"Trades:   {trades}",
        "",
        "Status: SHADOW MODE",
        "(500 trades before full)",
    ]
    buttons = [ModalButton("ENTER", "OK", "dismiss")]
    modal.show(
        title="✅ STRATEGY FORGED",
        body=body,
        buttons=buttons,
        style="bright_green",
        border_style="bright_green",
    )
    return modal


def build_forge_failure_modal(
    name: str,
    failed_at: str = "",
    reasons: list[str] | None = None,
    suggestions: list[str] | None = None,
) -> ModalState:
    """Build the forge failure modal."""
    modal = ModalState()
    body = [
        f"Name: {name}",
        f"Failed at: {failed_at}",
        "",
        "Reasons:",
    ]
    for r in (reasons or []):
        body.append(f"  • {r}")

    if suggestions:
        body.append("")
        body.append("Suggestions:")
        for s in suggestions:
            body.append(f"  • {s}")

    buttons = [
        ModalButton("ENTER", "OK", "dismiss"),
        ModalButton("V", "View Full Report", "view_report"),
    ]
    modal.show(
        title="✗ STRATEGY REJECTED",
        body=body,
        buttons=buttons,
        style="bright_red",
        border_style="bright_red",
    )
    return modal


def build_first_run_modal() -> ModalState:
    """Build the first-run wizard modal."""
    modal = ModalState()
    body = [
        "First run detected. Quick setup:",
        "",
        "1. Do you have MetaTrader 5 available?",
        "",
        "2. Starting capital: $10,000",
        "",
        "3. HYDRA checkpoint:",
    ]
    buttons = [
        ModalButton("Y", "Yes, set up MT5 connection", "mt5_setup"),
        ModalButton("N", "No, use simulated mode", "simulated"),
        ModalButton("T", "Train HYDRA from scratch", "train"),
        ModalButton("ENTER", "Continue with defaults", "defaults"),
    ]
    modal.show(
        title="Welcome to APHELION",
        body=body,
        buttons=buttons,
        style="bright_cyan",
        border_style="bright_cyan",
    )
    return modal
