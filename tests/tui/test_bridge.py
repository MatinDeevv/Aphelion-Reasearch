"""Tests for the TUI <-> PaperSession bridge."""

import pytest
from datetime import datetime, timezone

from aphelion.tui.state import TUIState, PositionView
from aphelion.tui.bridge import TUIBridge


class TestTUIBridge:
    """Unit tests for TUIBridge routing data into TUIState."""

    def _make_bridge(self) -> TUIBridge:
        return TUIBridge(TUIState())

    # ── update_bar ──────────────────────────────────────────────

    def test_update_bar_sets_fields(self):
        b = self._make_bridge()
        now = datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        b.update_bar(now, "LONDON", True, 42)
        s = b.state
        assert s.current_time == now
        assert s.current_session == "LONDON"
        assert s.market_open is True
        assert s.bars_processed == 42
        assert s.last_bar_time == now

    def test_update_bar_increments(self):
        b = self._make_bridge()
        t1 = datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc)
        t2 = datetime(2024, 6, 15, 14, 31, tzinfo=timezone.utc)
        b.update_bar(t1, "LONDON", True, 1)
        b.update_bar(t2, "LONDON", True, 2)
        assert b.state.bars_processed == 2
        assert b.state.last_bar_time == t2

    # ── update_hydra_signal ─────────────────────────────────────

    def test_update_hydra_signal_full(self):
        b = self._make_bridge()
        b.update_hydra_signal(
            direction="LONG",
            confidence=0.82,
            uncertainty=0.15,
            probs_5m=[0.1, 0.2, 0.7],
            probs_15m=[0.05, 0.15, 0.8],
            probs_1h=[0.1, 0.3, 0.6],
            horizon_agreement=0.9,
            gate_weights=[0.4, 0.3, 0.2, 0.1],
            moe_routing=[0.5, 0.2, 0.2, 0.1],
            top_features=[("rsi_14", 0.35), ("atr_14", 0.20)],
        )
        h = b.state.hydra
        assert h.direction == "LONG"
        assert h.confidence == pytest.approx(0.82)
        assert h.uncertainty == pytest.approx(0.15)
        assert h.probs_5m == [0.1, 0.2, 0.7]
        assert h.horizon_agreement == pytest.approx(0.9)
        assert len(h.gate_weights) == 4
        assert len(h.moe_routing) == 4
        assert h.top_features[0][0] == "rsi_14"
        assert h.timestamp is not None

    def test_update_hydra_signal_no_features(self):
        b = self._make_bridge()
        b.update_hydra_signal("SHORT", 0.5, 0.3,
                              [0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.3, 0.3],
                              0.6, [0.25]*4, [0.25]*4)
        assert b.state.hydra.top_features == []

    # ── update_sentinel ─────────────────────────────────────────

    def test_update_sentinel_normal(self):
        b = self._make_bridge()
        b.update_sentinel(False, False, False, 2, 0.04, 0.02, 100_500.0, True)
        s = b.state.sentinel
        assert s.trading_allowed is True
        assert s.open_positions == 2
        assert s.circuit_breaker_active is False

    def test_update_sentinel_l2_triggers_breaker(self):
        b = self._make_bridge()
        b.update_sentinel(True, True, False, 0, 0.0, 0.08, 98_000.0, False)
        s = b.state.sentinel
        assert s.l1_triggered is True
        assert s.l2_triggered is True
        assert s.circuit_breaker_active is True
        assert s.trading_allowed is False

    def test_update_sentinel_l3_triggers_breaker(self):
        b = self._make_bridge()
        b.update_sentinel(True, True, True, 0, 0.0, 0.12, 95_000.0, False)
        s = b.state.sentinel
        assert s.l3_triggered is True
        assert s.circuit_breaker_active is True

    # ── update_equity ───────────────────────────────────────────

    def test_update_equity_basic(self):
        b = self._make_bridge()
        b.update_equity(105_000.0, 500.0, 400.0, 100.0, 10, 7, 3)
        e = b.state.equity
        assert e.account_equity == 105_000.0
        assert e.daily_pnl == 500.0
        assert e.realized_pnl == 400.0
        assert e.unrealized_pnl == 100.0
        assert e.total_trades == 10
        assert e.winning_trades == 7
        assert e.losing_trades == 3

    def test_update_equity_session_peak_tracks_max(self):
        b = self._make_bridge()
        b.update_equity(105_000.0, 500.0, 400.0, 100.0, 5, 3, 2)
        b.update_equity(103_000.0, -200.0, 300.0, -100.0, 6, 3, 3)
        e = b.state.equity
        # peak should still be 105k even though current dropped
        assert e.session_peak == 105_000.0
        assert e.account_equity == 103_000.0

    def test_update_equity_negative_pnl(self):
        b = self._make_bridge()
        b.update_equity(98_000.0, -2000.0, -1500.0, -500.0, 5, 1, 4)
        e = b.state.equity
        assert e.daily_pnl == -2000.0
        assert e.account_equity == 98_000.0

    # ── update_positions ────────────────────────────────────────

    def test_update_positions_replaces(self):
        b = self._make_bridge()
        pos1 = [PositionView(position_id="p1", direction="LONG", entry_price=2850.0)]
        b.update_positions(pos1)
        assert len(b.state.positions) == 1

        pos2 = [
            PositionView(position_id="p2", direction="SHORT", entry_price=2870.0),
            PositionView(position_id="p3", direction="LONG", entry_price=2840.0),
        ]
        b.update_positions(pos2)
        assert len(b.state.positions) == 2
        assert b.state.positions[0].position_id == "p2"

    def test_update_positions_empty(self):
        b = self._make_bridge()
        b.update_positions([PositionView(position_id="p1")])
        b.update_positions([])
        assert len(b.state.positions) == 0

    # ── log ─────────────────────────────────────────────────────

    def test_log_pushes_entry(self):
        b = self._make_bridge()
        b.log("FILL", "Buy 0.01 XAUUSD @ 2850.00")
        assert len(b.state.log) == 1
        assert b.state.log[0].level == "FILL"
        assert "2850" in b.state.log[0].message

    def test_log_multiple_levels(self):
        b = self._make_bridge()
        b.log("INFO", "Session started")
        b.log("WARNING", "High volatility")
        b.log("ERROR", "Feed timeout")
        b.log("SENTINEL", "L1 triggered")
        assert len(b.state.log) == 4
        levels = [e.level for e in b.state.log]
        assert levels == ["INFO", "WARNING", "ERROR", "SENTINEL"]

    # ── state accessor ──────────────────────────────────────────

    def test_state_property(self):
        state = TUIState(session_name="Test-X")
        b = TUIBridge(state)
        assert b.state is state
        assert b.state.session_name == "Test-X"
