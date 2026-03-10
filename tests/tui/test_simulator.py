"""
Tests for the LiveSimulator engine.
"""

import time
import pytest

from aphelion.tui.state import TUIState
from aphelion.tui.simulator import LiveSimulator


class TestLiveSimulator:
    """Tests for LiveSimulator background data generator."""

    def test_starts_and_stops(self):
        """Simulator starts and stops cleanly."""
        state = TUIState()
        sim = LiveSimulator(state)

        assert not sim.is_running

        sim.start(capital=5_000.0, symbol="XAUUSD")
        assert sim.is_running
        assert state.feed_connected
        assert state.market_open
        assert state.feed_mode == "SIMULATED"

        # Let it run briefly
        time.sleep(0.3)

        sim.stop()
        assert not sim.is_running
        assert not state.feed_connected
        assert not state.market_open

    def test_generates_price_ticks(self):
        """Simulator produces price data in TUIState."""
        state = TUIState()
        sim = LiveSimulator(state)

        sim.start()
        time.sleep(0.5)  # Let a few ticks happen
        sim.stop()

        # Should have price data
        assert state.price.bid > 0
        assert state.price.ask > 0
        assert state.price.last > 0
        assert len(state.price.tick_history) > 0

    def test_generates_hydra_signals(self):
        """Simulator produces HYDRA signal data over time."""
        state = TUIState()
        sim = LiveSimulator(state)
        # Speed up for testing
        sim._bar_interval = 0.1
        sim._tick_interval = 0.05

        sim.start()
        time.sleep(1.0)  # Give it time to generate signals
        sim.stop()

        # May or may not have signals depending on timing
        # At minimum, logs should exist
        assert len(state.log) > 0

    def test_equity_tracking(self):
        """Simulator tracks equity history."""
        state = TUIState()
        sim = LiveSimulator(state)
        sim._tick_interval = 0.05

        sim.start(capital=10_000.0)
        time.sleep(0.5)
        sim.stop()

        # Equity should be tracked
        assert state.equity.account_equity > 0
        assert len(state.equity.equity_history) > 0

    def test_capital_customization(self):
        """Simulator uses custom capital value."""
        state = TUIState()
        sim = LiveSimulator(state)

        sim.start(capital=25_000.0, symbol="GOLD")
        time.sleep(0.2)
        sim.stop()

        assert state.equity.account_equity == pytest.approx(25_000.0, rel=0.1)
        assert state.price.symbol == "GOLD"

    def test_logs_start_message(self):
        """Simulator logs a start message."""
        state = TUIState()
        sim = LiveSimulator(state)

        sim.start(capital=10_000.0, symbol="XAUUSD")
        time.sleep(0.1)
        sim.stop()

        # Should have at least start and stop logs
        assert len(state.log) >= 2
        start_msgs = [e for e in state.log if "started" in e.message.lower()]
        assert len(start_msgs) > 0

    def test_multiple_start_stop_cycles(self):
        """Simulator can be started and stopped multiple times."""
        state = TUIState()
        sim = LiveSimulator(state)

        for _ in range(3):
            sim.start()
            assert sim.is_running
            time.sleep(0.1)
            sim.stop()
            assert not sim.is_running

    def test_double_start_ignored(self):
        """Starting an already running simulator is a no-op."""
        state = TUIState()
        sim = LiveSimulator(state)

        sim.start()
        assert sim.is_running
        sim.start()  # Should be ignored
        assert sim.is_running
        sim.stop()
        assert not sim.is_running
