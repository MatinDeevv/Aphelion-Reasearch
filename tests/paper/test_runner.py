"""Tests for PaperRunner — Phase 5 end-to-end orchestrator."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aphelion.core.config import Timeframe
from aphelion.core.data_layer import Bar
from aphelion.paper.feed import FeedConfig, FeedMode, SimulatedFeedConfig
from aphelion.paper.runner import PaperRunner, PaperRunnerConfig
from aphelion.paper.session import PaperSessionConfig, PaperSessionResult
from aphelion.risk.sentinel.execution.mt5 import MT5Config
from aphelion.tui.state import TUIState
from aphelion.tui.bridge import TUIBridge


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bar(ts_offset: int = 0, close: float = 2350.0) -> Bar:
    return Bar(
        timestamp=datetime.fromtimestamp(1704067200.0 + ts_offset * 60, tz=timezone.utc),
        timeframe=Timeframe.M1,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=500.0,
        tick_volume=500,
        spread=0.20,
        is_complete=True,
    )


def _sim_runner_config(max_bars: int = 50) -> PaperRunnerConfig:
    """Create a runner config for simulated mode with limited bars."""
    return PaperRunnerConfig(
        feed_mode=FeedMode.SIMULATED,
        session_config=PaperSessionConfig(
            initial_capital=10_000.0,
            warmup_bars=5,
        ),
        sim_config=SimulatedFeedConfig(max_bars=max_bars, seed=42),
        enable_tui=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER RUNNER CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaperRunnerConfig:
    """Tests for PaperRunnerConfig dataclass."""

    def test_defaults(self):
        """Default config should use SIMULATED mode."""
        cfg = PaperRunnerConfig()
        assert cfg.feed_mode == FeedMode.SIMULATED
        assert cfg.enable_tui is True
        assert cfg.session_config.initial_capital == 10_000.0

    def test_mt5_tick_config(self):
        """MT5 tick config should carry MT5 settings."""
        cfg = PaperRunnerConfig(
            feed_mode=FeedMode.MT5_TICK,
            mt5_config=MT5Config(login=12345, server="Eightcap-Demo"),
            feed_config=FeedConfig(poll_interval_ms=50),
        )
        assert cfg.feed_mode == FeedMode.MT5_TICK
        assert cfg.mt5_config.login == 12345
        assert cfg.feed_config.poll_interval_ms == 50


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER RUNNER EXECUTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaperRunner:
    """Tests for PaperRunner with simulated feed."""

    async def test_simulated_run_completes(self):
        """Runner should complete a simulated run and return results."""
        config = _sim_runner_config(max_bars=30)
        runner = PaperRunner(config)
        result = await runner.run()

        assert isinstance(result, PaperSessionResult)
        assert result.bars_processed == 30
        assert result.initial_capital == 10_000.0

    async def test_result_has_session_id(self):
        """Result should carry the session ID from config."""
        config = _sim_runner_config(max_bars=10)
        runner = PaperRunner(config)
        result = await runner.run()

        assert result.session_id == config.session_config.session_id

    async def test_equity_preserved_without_trades(self):
        """Without HYDRA, equity should stay near initial capital."""
        config = _sim_runner_config(max_bars=20)
        runner = PaperRunner(config)
        result = await runner.run()

        # No HYDRA checkpoint → no trades
        assert result.total_trades == 0
        assert result.final_equity == config.session_config.initial_capital

    async def test_stop_signal(self):
        """stop() should end the session gracefully."""
        config = _sim_runner_config(max_bars=0)  # Infinite
        runner = PaperRunner(config)

        async def stop_after_delay():
            await asyncio.sleep(0.3)
            await runner.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        result = await runner.run()
        await stop_task

        assert isinstance(result, PaperSessionResult)
        assert result.bars_processed >= 1

    async def test_tui_state_created_when_enabled(self):
        """When TUI is enabled, runner should create TUIState."""
        config = _sim_runner_config(max_bars=10)
        config.enable_tui = True
        runner = PaperRunner(config)

        result = await runner.run()

        assert runner._tui_state is not None
        assert isinstance(runner._tui_state, TUIState)
        assert runner._tui_bridge is not None

    async def test_no_tui_when_disabled(self):
        """When TUI is disabled, state and bridge should be None."""
        config = _sim_runner_config(max_bars=10)
        config.enable_tui = False
        runner = PaperRunner(config)

        result = await runner.run()

        assert runner._tui_state is None
        assert runner._tui_bridge is None

    async def test_replay_mode(self):
        """Runner should work with replay feed mode."""
        bars = [_make_bar(ts_offset=i, close=2350.0 + i * 0.1) for i in range(25)]
        config = PaperRunnerConfig(
            feed_mode=FeedMode.REPLAY,
            session_config=PaperSessionConfig(
                initial_capital=10_000.0,
                warmup_bars=5,
            ),
            replay_bars=bars,
            enable_tui=False,
        )
        runner = PaperRunner(config)
        result = await runner.run()

        assert result.bars_processed == 25

    async def test_mt5_tick_fails_gracefully(self):
        """Runner should raise ConnectionError if MT5 cannot connect."""
        config = PaperRunnerConfig(
            feed_mode=FeedMode.MT5_TICK,
            mt5_config=MT5Config(),
            feed_config=FeedConfig(
                warmup_bars=0,
                reconnect_delay_s=0.01,
                max_reconnect_attempts=1,
            ),
            enable_tui=False,
        )

        # Mock MT5Connection to fail
        with patch("aphelion.paper.runner.MT5Connection") as MockConn:
            mock_conn = MagicMock()
            mock_conn.connect.return_value = False
            mock_conn.is_connected = False
            MockConn.return_value = mock_conn

            runner = PaperRunner(config)
            with pytest.raises(ConnectionError):
                await runner.run()
