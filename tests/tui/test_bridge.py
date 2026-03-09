"""Tests for the TUI <-> PaperSession bridge (v2 - Bloomberg-grade)."""

import pytest
from datetime import datetime, timezone

from aphelion.tui.state import TUIState, PositionView
from aphelion.tui.bridge import TUIBridge


class TestTUIBridge:
    """Unit tests for TUIBridge routing data into TUIState."""

    def _make_bridge(self) -> TUIBridge:
        return TUIBridge(TUIState())

    # -- update_bar ---------------------------------------------------

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

    # -- update_hydra_signal ------------------------------------------

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

    def test_update_hydra_confidence_history(self):
        b = self._make_bridge()
        for conf in [0.3, 0.5, 0.7, 0.9]:
            b.update_hydra_signal(
                "LONG", conf, 0.1,
                [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7],
                0.8, [0.25]*4, [0.25]*4,
            )
        assert len(b.state.hydra.confidence_history) == 4
        assert b.state.hydra.signal_count == 4

    def test_update_hydra_signal_no_features(self):
        b = self._make_bridge()
        b.update_hydra_signal("SHORT", 0.5, 0.3,
                              [0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.3, 0.3],
                              0.6, [0.25]*4, [0.25]*4)
        assert b.state.hydra.top_features == []

    # -- update_sentinel ----------------------------------------------

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

    def test_update_sentinel_breaker_alert(self):
        b = self._make_bridge()
        b.update_sentinel(False, False, False, 0, 0.0, 0.01, 100_000.0, True)
        assert len(b.state.alerts) == 0
        b.update_sentinel(True, True, False, 0, 0.0, 0.05, 98_000.0, False)
        assert len(b.state.alerts) == 1
        assert b.state.alerts[0].severity == "CRITICAL"
        assert b.state.sentinel.breaker_since is not None

    def test_update_sentinel_breaker_clear_alert(self):
        b = self._make_bridge()
        b.update_sentinel(True, False, False, 0, 0.0, 0.03, 99_000.0, False)
        b.update_sentinel(False, False, False, 0, 0.0, 0.01, 100_000.0, True)
        assert len(b.state.alerts) == 2
        assert b.state.alerts[1].severity == "INFO"
        assert b.state.sentinel.breaker_since is None

    def test_update_sentinel_drawdown_history(self):
        b = self._make_bridge()
        for dd in [0.01, 0.02, 0.015]:
            b.update_sentinel(False, False, False, 0, 0.0, dd, 100_000.0, True)
        assert len(b.state.sentinel.drawdown_history) == 3

    # -- update_equity ------------------------------------------------

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
        assert e.session_peak == 105_000.0
        assert e.account_equity == 103_000.0

    def test_update_equity_records_history(self):
        b = self._make_bridge()
        b.update_equity(100_000.0, 0.0, 0.0, 0.0, 0, 0, 0)
        b.update_equity(101_000.0, 1000.0, 500.0, 500.0, 1, 1, 0)
        assert len(b.state.equity.equity_history) == 2

    def test_update_equity_negative_pnl(self):
        b = self._make_bridge()
        b.update_equity(98_000.0, -2000.0, -1500.0, -500.0, 5, 1, 4)
        e = b.state.equity
        assert e.daily_pnl == -2000.0
        assert e.account_equity == 98_000.0

    # -- update_positions ---------------------------------------------

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

    # -- update_price (v2) --------------------------------------------

    def test_update_price(self):
        b = self._make_bridge()
        b.update_price(
            bid=2350.0, ask=2350.5,
            change=12.0, change_pct=0.51,
            high=2365.0, low=2340.0,
        )
        p = b.state.price
        assert p.bid == 2350.0
        assert p.ask == 2350.5
        assert p.change == 12.0
        assert p.high == 2365.0
        assert len(p.tick_history) == 1

    def test_update_price_throttled(self):
        b = TUIBridge(TUIState(), min_update_interval=1.0)
        b.update_price(2350.0, 2350.5)
        b.update_price(2351.0, 2351.5)
        # Second call should be throttled
        assert b.state.price.bid == 2350.0

    # -- update_performance (v2) --------------------------------------

    def test_update_performance(self):
        b = self._make_bridge()
        b.update_performance(
            sharpe_ratio=2.1,
            profit_factor=1.8,
            max_drawdown_pct=0.02,
            avg_win=120.0,
            avg_loss=-80.0,
            best_trade=350.0,
            worst_trade=-200.0,
            avg_hold_bars=12.5,
            consecutive_wins=3,
            consecutive_losses=0,
        )
        e = b.state.equity
        assert e.sharpe_ratio == pytest.approx(2.1)
        assert e.profit_factor == pytest.approx(1.8)
        assert e.best_trade == 350.0
        assert e.consecutive_wins == 3

    # -- update_system_stats (v2) -------------------------------------

    def test_update_system_stats(self):
        b = self._make_bridge()
        b.update_system_stats(cpu_usage=45.0, memory_mb=512.0, latency_ms=15.0, uptime_seconds=3600.0)
        assert b.state.cpu_usage == pytest.approx(45.0)
        assert b.state.memory_mb == pytest.approx(512.0)

    # -- log and alert ------------------------------------------------

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

    def test_alert_method(self):
        b = self._make_bridge()
        b.alert("CRITICAL", "High DD", "Drawdown > 4%")
        assert len(b.state.alerts) == 1
        assert b.state.alerts[0].severity == "CRITICAL"
        assert b.state.alerts[0].title == "High DD"

    # -- state accessor -----------------------------------------------

    def test_state_property(self):
        state = TUIState(session_name="Test-X")
        b = TUIBridge(state)
        assert b.state is state
        assert b.state.session_name == "Test-X"

    def test_min_update_interval_param(self):
        b = TUIBridge(TUIState(), min_update_interval=0.5)
        assert b._min_interval == 0.5
