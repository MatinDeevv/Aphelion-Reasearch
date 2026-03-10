"""Tests for TITAN validators and reporter."""

import pytest

from aphelion.risk.titan.gate import ValidationResult, GateReport, GateStatus
from aphelion.risk.titan.reporter import TitanReporter
from aphelion.risk.titan.validators.performance import PerformanceValidator
from aphelion.risk.titan.validators.stability import StabilityValidator
from aphelion.risk.titan.validators.stress import StressValidator
from aphelion.risk.titan.validators.regression import RegressionValidator
from aphelion.risk.titan.validators.latency import LatencyValidator


# ── PerformanceValidator ────────────────────────────────────────────────────

class TestPerformanceValidator:

    def test_pass_all(self):
        v = PerformanceValidator()
        results = v.validate(sharpe=1.8, win_rate=0.58, max_drawdown=0.04,
                             profit_factor=1.5, total_trades=300)
        assert all(r.passed for r in results)

    def test_fail_win_rate(self):
        v = PerformanceValidator()
        results = v.validate(sharpe=1.8, win_rate=0.40, max_drawdown=0.04,
                             profit_factor=1.5, total_trades=300)
        failed = [r for r in results if not r.passed and "win" in r.check_name.lower()]
        assert len(failed) >= 1

    def test_fail_sharpe(self):
        v = PerformanceValidator()
        results = v.validate(sharpe=0.5, win_rate=0.58, max_drawdown=0.04,
                             profit_factor=1.5, total_trades=300)
        failed = [r for r in results if not r.passed and "sharpe" in r.check_name.lower()]
        assert len(failed) >= 1

    def test_fail_drawdown(self):
        v = PerformanceValidator()
        results = v.validate(sharpe=1.8, win_rate=0.58, max_drawdown=0.20,
                             profit_factor=1.5, total_trades=300)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1

    def test_result_structure(self):
        v = PerformanceValidator()
        results = v.validate(sharpe=1.8, win_rate=0.58, max_drawdown=0.04,
                             profit_factor=1.5, total_trades=300)
        for r in results:
            assert isinstance(r, ValidationResult)
            assert isinstance(r.passed, bool)


# ── StabilityValidator ──────────────────────────────────────────────────────

class TestStabilityValidator:

    def test_pass(self):
        v = StabilityValidator()
        results = v.validate(
            fold_sharpes=[1.5, 1.4, 1.6, 1.3, 1.7, 1.5, 1.4, 1.6, 1.5, 1.4],
        )
        assert all(r.passed for r in results)

    def test_fail_low_sharpes(self):
        v = StabilityValidator()
        results = v.validate(
            fold_sharpes=[0.3, 0.2, 0.4, 0.1, -0.5, 0.3, 0.2, 0.1, -0.2, 0.3],
        )
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1


# ── StressValidator ─────────────────────────────────────────────────────────

class TestStressValidator:

    def test_pass(self):
        v = StressValidator()
        results = v.validate(
            mc_p5_sharpe=1.0,
            mc_p1_drawdown=0.10,
            stress_scenarios_passed=4,
            stress_scenarios_total=5,
        )
        assert all(r.passed for r in results)

    def test_fail_mc_sharpe(self):
        v = StressValidator()
        results = v.validate(
            mc_p5_sharpe=0.3,
            mc_p1_drawdown=0.10,
            stress_scenarios_passed=4,
            stress_scenarios_total=5,
        )
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1

    def test_fail_stress_pass_rate(self):
        v = StressValidator()
        results = v.validate(
            mc_p5_sharpe=1.0,
            mc_p1_drawdown=0.10,
            stress_scenarios_passed=1,
            stress_scenarios_total=5,
        )
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1


# ── RegressionValidator ─────────────────────────────────────────────────────

class TestRegressionValidator:

    def test_pass(self):
        v = RegressionValidator()
        baseline = {"sharpe": 1.5, "win_rate": 0.55, "profit_factor": 1.3}
        current = {"sharpe": 1.5, "win_rate": 0.56, "profit_factor": 1.4}
        results = v.validate(baseline, current)
        assert all(r.passed for r in results)

    def test_fail_regression(self):
        v = RegressionValidator()
        baseline = {"sharpe": 1.5, "win_rate": 0.55, "profit_factor": 1.3}
        current = {"sharpe": 0.8, "win_rate": 0.40, "profit_factor": 0.8}
        results = v.validate(baseline, current, max_regression_pct=0.10)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1


# ── LatencyValidator ────────────────────────────────────────────────────────

class TestLatencyValidator:

    def test_pass(self):
        v = LatencyValidator()
        buckets = {"feature_engine": 50.0, "hydra": 80.0, "ares": 30.0}
        results = v.validate(buckets)
        assert all(r.passed for r in results)

    def test_fail_single_p99(self):
        v = LatencyValidator()
        buckets = {"feature_engine": 50.0, "hydra": 300.0, "ares": 30.0}
        results = v.validate(buckets, max_single_p99_ms=200.0)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1

    def test_fail_total(self):
        v = LatencyValidator()
        buckets = {"a": 200.0, "b": 200.0, "c": 200.0}
        results = v.validate(buckets, max_total_p99_ms=500.0)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1


# ── TitanReporter ───────────────────────────────────────────────────────────

class TestTitanReporter:

    def _make_report(self, passed=True):
        status = GateStatus.PASSED if passed else GateStatus.FAILED
        return GateReport(
            status=status,
            triggered_by="test",
            validations=[
                ValidationResult("sharpe", passed, 1.5, 1.5),
            ],
            failures=[] if passed else ["Sharpe below threshold"],
        )

    def test_text_report(self):
        reporter = TitanReporter()
        report = self._make_report(True)
        reporter.add_report(report)
        text = reporter.generate_text(report)
        assert "PASSED" in text

    def test_json_report(self):
        reporter = TitanReporter()
        report = self._make_report(True)
        reporter.add_report(report)
        data = reporter.generate_json(report)
        assert data["status"] == "PASSED"

    def test_failed_report(self):
        reporter = TitanReporter()
        report = self._make_report(False)
        reporter.add_report(report)
        text = reporter.generate_text(report)
        assert "FAIL" in text or "fail" in text.lower()

    def test_total_reports(self):
        reporter = TitanReporter()
        reporter.add_report(self._make_report(True))
        reporter.add_report(self._make_report(False))
        assert reporter.total_reports == 2

    def test_latest_status(self):
        reporter = TitanReporter()
        reporter.add_report(self._make_report(True))
        assert reporter.latest_status == GateStatus.PASSED
