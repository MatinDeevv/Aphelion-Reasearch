"""
Tests for APHELION TUI v1 (Phase 6).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from aphelion.tui.state import (
    TUIState,
    HydraSignalView,
    SentinelView,
    EquityView,
    PositionView,
    LogEntry,
)
from aphelion.tui.app import AphelionTUI, TUIConfig

# ── Rich is required for Phase 6 ────────────────────────────────────────────
rich = pytest.importorskip("rich")

from aphelion.tui.screens.header import build_header
from aphelion.tui.screens.hydra_panel import build_hydra_panel
from aphelion.tui.screens.sentinel_panel import build_sentinel_panel
from aphelion.tui.screens.positions import build_positions_panel
from aphelion.tui.screens.equity import build_equity_panel
from aphelion.tui.screens.event_log import build_log_panel
from aphelion.tui.screens.dashboard import build_dashboard_layout

from rich.panel import Panel
from rich.layout import Layout


# ═══════════════════════════════════════════════════════════════════════════
# State dataclass tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTUIState:
    """TUIState dataclass behaviour."""

    def test_default_creation(self):
        state = TUIState()
        assert state.session_name == "Paper-01"
        assert state.bars_processed == 0
        assert state.market_open is False
        assert isinstance(state.hydra, HydraSignalView)
        assert isinstance(state.sentinel, SentinelView)
        assert isinstance(state.equity, EquityView)
        assert state.positions == []
        assert state.log == []

    def test_push_log(self):
        state = TUIState(max_log_lines=5)
        for i in range(10):
            state.push_log("INFO", f"msg-{i}")
        assert len(state.log) == 5
        assert state.log[0].message == "msg-5"
        assert state.log[-1].message == "msg-9"

    def test_push_log_levels(self):
        state = TUIState()
        state.push_log("FILL", "Order filled LONG 0.10 lots")
        state.push_log("REJECT", "SENTINEL blocked: max exposure")
        state.push_log("ERROR", "Connection lost")
        assert state.log[0].level == "FILL"
        assert state.log[1].level == "REJECT"
        assert state.log[2].level == "ERROR"

    def test_hydra_signal_defaults(self):
        h = HydraSignalView()
        assert h.direction == "FLAT"
        assert h.confidence == 0.0
        assert len(h.probs_5m) == 3
        assert len(h.gate_weights) == 4
        assert len(h.moe_routing) == 4

    def test_sentinel_view_defaults(self):
        s = SentinelView()
        assert s.trading_allowed is True
        assert s.l3_triggered is False
        assert s.max_positions == 3

    def test_equity_view_defaults(self):
        e = EquityView()
        assert e.account_equity == 100_000.0
        assert e.total_trades == 0

    def test_position_view_fields(self):
        p = PositionView(
            position_id="P001",
            direction="LONG",
            entry_price=2350.0,
            current_price=2355.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            size_lots=0.10,
            unrealized_pnl=50.0,
        )
        assert p.direction == "LONG"
        assert p.unrealized_pnl == 50.0


# ═══════════════════════════════════════════════════════════════════════════
# Screen builder tests (each must return Rich renderables without error)
# ═══════════════════════════════════════════════════════════════════════════

class TestScreenBuilders:
    """Each screen builder produces a valid Rich Panel / Layout."""

    @pytest.fixture
    def state(self):
        return TUIState(
            session_name="Test-Session",
            market_open=True,
            current_session="LONDON",
            current_time=datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            bars_processed=42,
        )

    def test_header_returns_panel(self, state):
        panel = build_header(state)
        assert isinstance(panel, Panel)

    def test_hydra_panel_returns_panel(self, state):
        state.hydra.direction = "LONG"
        state.hydra.confidence = 0.78
        state.hydra.uncertainty = 0.12
        state.hydra.horizon_agreement = 1.0
        state.hydra.probs_1h = [0.1, 0.12, 0.78]
        state.hydra.gate_weights = [0.4, 0.3, 0.2, 0.1]
        state.hydra.moe_routing = [0.5, 0.2, 0.2, 0.1]
        state.hydra.top_features = [("vpin", 0.15), ("atr", 0.12)]
        panel = build_hydra_panel(state)
        assert isinstance(panel, Panel)

    def test_hydra_panel_flat(self, state):
        """Flat signal still renders."""
        panel = build_hydra_panel(state)
        assert isinstance(panel, Panel)

    def test_sentinel_panel_returns_panel(self, state):
        panel = build_sentinel_panel(state)
        assert isinstance(panel, Panel)

    def test_sentinel_panel_with_triggers(self, state):
        state.sentinel.l1_triggered = True
        state.sentinel.l3_triggered = True
        state.sentinel.trading_allowed = False
        panel = build_sentinel_panel(state)
        assert isinstance(panel, Panel)

    def test_positions_panel_empty(self, state):
        panel = build_positions_panel(state)
        assert isinstance(panel, Panel)

    def test_positions_panel_with_data(self, state):
        state.positions = [
            PositionView(
                position_id="P001",
                direction="LONG",
                entry_price=2350.0,
                current_price=2358.0,
                stop_loss=2340.0,
                take_profit=2370.0,
                size_lots=0.10,
                unrealized_pnl=80.0,
            ),
            PositionView(
                position_id="P002",
                direction="SHORT",
                entry_price=2360.0,
                current_price=2362.0,
                stop_loss=2370.0,
                take_profit=2350.0,
                size_lots=0.05,
                unrealized_pnl=-10.0,
            ),
        ]
        panel = build_positions_panel(state)
        assert isinstance(panel, Panel)

    def test_equity_panel_returns_panel(self, state):
        state.equity.daily_pnl = 150.0
        state.equity.total_trades = 5
        state.equity.winning_trades = 3
        state.equity.losing_trades = 2
        panel = build_equity_panel(state)
        assert isinstance(panel, Panel)

    def test_equity_panel_negative(self, state):
        state.equity.daily_pnl = -200.0
        panel = build_equity_panel(state)
        assert isinstance(panel, Panel)

    def test_log_panel_empty(self, state):
        panel = build_log_panel(state)
        assert isinstance(panel, Panel)

    def test_log_panel_with_entries(self, state):
        state.push_log("FILL", "Filled LONG 0.10 at 2350.50")
        state.push_log("INFO", "Bar 43 processed")
        state.push_log("SENTINEL", "Exposure check passed")
        panel = build_log_panel(state)
        assert isinstance(panel, Panel)


# ═══════════════════════════════════════════════════════════════════════════
# Full dashboard compositor
# ═══════════════════════════════════════════════════════════════════════════

class TestDashboardLayout:
    """Full dashboard layout integration."""

    def test_build_layout_returns_layout(self):
        state = TUIState()
        layout = build_dashboard_layout(state)
        assert isinstance(layout, Layout)

    def test_layout_with_full_state(self):
        state = TUIState(
            market_open=True,
            bars_processed=100,
        )
        state.hydra.direction = "SHORT"
        state.hydra.confidence = 0.65
        state.sentinel.l1_triggered = True
        state.equity.daily_pnl = -50.0
        state.positions.append(PositionView(
            position_id="P123",
            direction="SHORT",
            entry_price=2400.0,
            current_price=2395.0,
            size_lots=0.10,
            unrealized_pnl=50.0,
        ))
        state.push_log("FILL", "Filled SHORT at 2400.0")
        layout = build_dashboard_layout(state)
        assert isinstance(layout, Layout)


# ═══════════════════════════════════════════════════════════════════════════
# AphelionTUI class
# ═══════════════════════════════════════════════════════════════════════════

class TestAphelionTUI:
    """TUI application class tests."""

    def test_creation(self):
        tui = AphelionTUI()
        assert tui.state is not None
        assert isinstance(tui.state, TUIState)

    def test_creation_with_state(self):
        state = TUIState(session_name="Custom")
        tui = AphelionTUI(state=state)
        assert tui.state.session_name == "Custom"

    def test_creation_with_config(self):
        cfg = TUIConfig(refresh_rate=1.0, compact_mode=True)
        tui = AphelionTUI(config=cfg)
        assert tui._config.compact_mode is True

    def test_stop_flag(self):
        tui = AphelionTUI()
        assert tui._running is False
        tui.stop()
        assert tui._running is False

    async def test_run_stops_quickly(self):
        """TUI should stop when stop() is called from another task."""
        tui = AphelionTUI(config=TUIConfig(refresh_rate=0.05))

        async def stopper():
            await asyncio.sleep(0.15)
            tui.stop()

        # Don't actually run with Live (needs a real terminal), just confirm logic
        # by testing the stop mechanism
        tui._running = True
        task = asyncio.create_task(stopper())
        # Simulate a mini loop
        iterations = 0
        while tui._running and iterations < 20:
            iterations += 1
            await asyncio.sleep(0.05)
        await task
        assert tui._running is False
        assert iterations < 20  # Stopped within reasonable time


# ═══════════════════════════════════════════════════════════════════════════
# Rendering smoke test — ensure Rich can render full layout to string
# ═══════════════════════════════════════════════════════════════════════════

class TestRenderSmoke:
    """Actually render the dashboard to a string buffer."""

    def test_render_to_string(self):
        """Full dashboard renders to a string without exceptions."""
        from rich.console import Console
        from io import StringIO

        state = TUIState(
            market_open=True,
            bars_processed=50,
            current_session="OVERLAP_LDN_NY",
            current_time=datetime(2025, 3, 10, 14, 0, 0, tzinfo=timezone.utc),
        )
        state.hydra.direction = "LONG"
        state.hydra.confidence = 0.82
        state.equity.daily_pnl = 320.50
        state.push_log("INFO", "Session started")

        buffer = StringIO()
        console = Console(file=buffer, width=120, force_terminal=True)
        layout = build_dashboard_layout(state)
        console.print(layout)
        output = buffer.getvalue()
        assert len(output) > 100  # Reasonably long output
        assert "APHELION" in output
