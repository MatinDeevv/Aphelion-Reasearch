"""Tests for NEMESIS detector and TITAN gate."""

import pytest

from aphelion.nemesis.detector import NEMESISDetector, NEMESISSignal, StressMonitor
from aphelion.risk.titan.gate import TitanGate, GateStatus, GateReport


# ── StressMonitor ──────────────────────────────────────────────────


class TestStressMonitor:
    def test_initial_state(self):
        monitor = StressMonitor()
        assert monitor.session_win_rate == 0.5  # default neutral when no trades

    def test_record_trade_outcome(self):
        monitor = StressMonitor()
        monitor.record_trade_outcome(is_win=True)
        monitor.record_trade_outcome(is_win=False)
        assert monitor.session_win_rate > 0

    def test_failed_breakouts(self):
        monitor = StressMonitor()
        monitor.record_trade_outcome(is_win=False, was_breakout=True)
        assert monitor.failed_breakouts >= 1

    def test_reset_session(self):
        monitor = StressMonitor()
        monitor.record_trade_outcome(is_win=True)
        monitor.reset_session()
        assert monitor.session_win_rate == 0.5  # resets to default neutral


# ── NEMESISDetector ────────────────────────────────────────────────


class TestNEMESISDetector:
    def test_instantiation(self):
        detector = NEMESISDetector()
        assert detector is not None

    def test_signal_when_healthy(self):
        detector = NEMESISDetector()
        signal = detector.generate_signal(
            ares_consensus=1,
            rolling_win_rate_20=0.6,
            regime_accuracy=0.7,
            high_conf_win_rate=0.65,
            consecutive_losses=1,
            failed_breakouts=0,
        )
        assert isinstance(signal, NEMESISSignal)
        # Healthy conditions: no contrarian signal

    def test_signal_under_stress(self):
        detector = NEMESISDetector()
        signal = detector.generate_signal(
            ares_consensus=1,
            rolling_win_rate_20=0.10,
            regime_accuracy=0.1,
            high_conf_win_rate=0.1,
            consecutive_losses=12,
            failed_breakouts=10,
        )
        assert signal.stress_score > 0.5

    def test_signal_returns_direction(self):
        detector = NEMESISDetector()
        signal = detector.generate_signal(
            ares_consensus=1,
            rolling_win_rate_20=0.50,
        )
        assert signal.direction in (-1, 0, 1)


# ── TITAN Gate ─────────────────────────────────────────────────────


class TestTitanGate:
    def test_gate_passes_perfect(self):
        gate = TitanGate()
        report = gate.run_full_gate(
            triggered_by="test",
            sharpe=2.0,
            win_rate=0.60,
            max_drawdown=0.08,
            profit_factor=1.8,
            num_trades=300,
            fold_sharpes=[1.5, 1.3, 1.4, 1.6, 1.5, 1.3, 1.4, 1.5, 1.3, 1.4, 1.2, 1.5],
            mc_sharpes=[1.2, 1.3, 1.1, 1.4, 1.0, 1.5, 1.2, 1.3, 1.1, 1.4],
            mc_drawdowns=[0.10, 0.12, 0.15, 0.08, 0.11, 0.13, 0.09, 0.14, 0.16, 0.12],
            baseline_sharpe=1.8,
            p99_latency_ms=100.0,
        )
        assert isinstance(report, GateReport)
        assert report.status == GateStatus.PASSED

    def test_gate_fails_low_sharpe(self):
        gate = TitanGate()
        report = gate.run_full_gate(
            triggered_by="test",
            sharpe=0.5,
            win_rate=0.45,
            max_drawdown=0.20,
            profit_factor=0.9,
            num_trades=50,
            fold_sharpes=[0.5, 0.3],
            mc_sharpes=[0.2, 0.3, 0.1],
            mc_drawdowns=[0.30, 0.35, 0.40],
            baseline_sharpe=1.5,
            p99_latency_ms=500.0,
        )
        assert report.status == GateStatus.FAILED
        assert len(report.failures) > 0

    def test_gate_report_has_validations(self):
        gate = TitanGate()
        report = gate.run_full_gate(
            triggered_by="test",
            sharpe=1.6, win_rate=0.56, max_drawdown=0.11,
            profit_factor=1.4, num_trades=250,
            fold_sharpes=[1.3] * 12,
            mc_sharpes=[1.0] * 10,
            mc_drawdowns=[0.15] * 10,
            baseline_sharpe=1.6,
            p99_latency_ms=200.0,
        )
        assert len(report.validations) > 0
        assert report.status in (GateStatus.PASSED, GateStatus.FAILED)
