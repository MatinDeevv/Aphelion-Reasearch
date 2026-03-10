"""
TITAN — System-Wide Quality Gate
Phase 15 — Engineering Spec v3.0

Nothing goes to production without TITAN's approval.
Runs performance, stability, stress, regression, and latency checks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum


class GateStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    actual_value: float
    threshold: float
    message: str = ""


@dataclass
class GateReport:
    status: GateStatus
    triggered_by: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validations: List[ValidationResult] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        if not self.validations:
            return 0.0
        return sum(1 for v in self.validations if v.passed) / len(self.validations)


# Quality gate criteria from spec
TITAN_REQUIREMENTS = {
    # Performance
    "min_sharpe_ratio": 1.5,
    "min_win_rate": 0.55,
    "max_drawdown": 0.12,
    "min_profit_factor": 1.3,
    "min_trades_for_significance": 200,

    # Stability (walk-forward)
    "wf_min_folds_passing": 8,
    "wf_min_median_sharpe": 1.2,
    "wf_max_sharpe_variance": 0.8,

    # Stress (Monte Carlo)
    "mc_5th_percentile_sharpe": 0.8,
    "mc_95th_percentile_max_dd": 0.25,

    # Regression
    "max_performance_regression": -0.10,

    # Latency
    "max_p99_latency_ms": 250,
}


class PerformanceValidator:
    """Validate performance metrics meet thresholds."""

    def validate(
        self,
        sharpe: float,
        win_rate: float,
        max_drawdown: float,
        profit_factor: float,
        num_trades: int,
    ) -> List[ValidationResult]:
        results = []

        results.append(ValidationResult(
            "min_sharpe_ratio", sharpe >= TITAN_REQUIREMENTS["min_sharpe_ratio"],
            sharpe, TITAN_REQUIREMENTS["min_sharpe_ratio"],
        ))
        results.append(ValidationResult(
            "min_win_rate", win_rate >= TITAN_REQUIREMENTS["min_win_rate"],
            win_rate, TITAN_REQUIREMENTS["min_win_rate"],
        ))
        results.append(ValidationResult(
            "max_drawdown", max_drawdown <= TITAN_REQUIREMENTS["max_drawdown"],
            max_drawdown, TITAN_REQUIREMENTS["max_drawdown"],
        ))
        results.append(ValidationResult(
            "min_profit_factor", profit_factor >= TITAN_REQUIREMENTS["min_profit_factor"],
            profit_factor, TITAN_REQUIREMENTS["min_profit_factor"],
        ))
        results.append(ValidationResult(
            "min_trades", num_trades >= TITAN_REQUIREMENTS["min_trades_for_significance"],
            num_trades, TITAN_REQUIREMENTS["min_trades_for_significance"],
        ))

        return results


class StabilityValidator:
    """Validate walk-forward consistency."""

    def validate(
        self,
        fold_sharpes: List[float],
        min_folds: int = 12,
    ) -> List[ValidationResult]:
        import numpy as np

        results = []
        passing_folds = sum(1 for s in fold_sharpes if s >= 1.0)
        results.append(ValidationResult(
            "wf_min_folds_passing",
            passing_folds >= TITAN_REQUIREMENTS["wf_min_folds_passing"],
            passing_folds, TITAN_REQUIREMENTS["wf_min_folds_passing"],
        ))

        if fold_sharpes:
            median_sharpe = float(np.median(fold_sharpes))
            sharpe_var = float(np.var(fold_sharpes))
        else:
            median_sharpe = 0.0
            sharpe_var = 999.0

        results.append(ValidationResult(
            "wf_min_median_sharpe",
            median_sharpe >= TITAN_REQUIREMENTS["wf_min_median_sharpe"],
            median_sharpe, TITAN_REQUIREMENTS["wf_min_median_sharpe"],
        ))
        results.append(ValidationResult(
            "wf_max_sharpe_variance",
            sharpe_var <= TITAN_REQUIREMENTS["wf_max_sharpe_variance"],
            sharpe_var, TITAN_REQUIREMENTS["wf_max_sharpe_variance"],
        ))

        return results


class StressValidator:
    """Validate Monte Carlo stress test results."""

    def validate(
        self,
        mc_sharpes: List[float],
        mc_drawdowns: List[float],
    ) -> List[ValidationResult]:
        import numpy as np

        results = []
        if mc_sharpes:
            p5_sharpe = float(np.percentile(mc_sharpes, 5))
        else:
            p5_sharpe = 0.0

        if mc_drawdowns:
            p95_dd = float(np.percentile(mc_drawdowns, 95))
        else:
            p95_dd = 1.0

        results.append(ValidationResult(
            "mc_5th_percentile_sharpe",
            p5_sharpe >= TITAN_REQUIREMENTS["mc_5th_percentile_sharpe"],
            p5_sharpe, TITAN_REQUIREMENTS["mc_5th_percentile_sharpe"],
        ))
        results.append(ValidationResult(
            "mc_95th_percentile_max_dd",
            p95_dd <= TITAN_REQUIREMENTS["mc_95th_percentile_max_dd"],
            p95_dd, TITAN_REQUIREMENTS["mc_95th_percentile_max_dd"],
        ))

        return results


class RegressionValidator:
    """Check for performance regression vs previous baseline."""

    def validate(
        self, current_sharpe: float, baseline_sharpe: float
    ) -> List[ValidationResult]:
        if baseline_sharpe > 0:
            regression = (current_sharpe - baseline_sharpe) / baseline_sharpe
        else:
            regression = 0.0

        return [ValidationResult(
            "max_performance_regression",
            regression >= TITAN_REQUIREMENTS["max_performance_regression"],
            regression, TITAN_REQUIREMENTS["max_performance_regression"],
        )]


class LatencyValidator:
    """Validate system latency."""

    def validate(self, p99_latency_ms: float) -> List[ValidationResult]:
        return [ValidationResult(
            "max_p99_latency_ms",
            p99_latency_ms <= TITAN_REQUIREMENTS["max_p99_latency_ms"],
            p99_latency_ms, TITAN_REQUIREMENTS["max_p99_latency_ms"],
        )]


class TitanGate:
    """
    Main TITAN quality gate orchestrator.
    Runs all validators and produces a GateReport.
    """

    def __init__(self):
        self._perf = PerformanceValidator()
        self._stability = StabilityValidator()
        self._stress = StressValidator()
        self._regression = RegressionValidator()
        self._latency = LatencyValidator()

    def run_full_gate(
        self,
        triggered_by: str = "manual",
        # Performance
        sharpe: float = 0.0,
        win_rate: float = 0.0,
        max_drawdown: float = 1.0,
        profit_factor: float = 0.0,
        num_trades: int = 0,
        # Stability
        fold_sharpes: Optional[List[float]] = None,
        # Stress
        mc_sharpes: Optional[List[float]] = None,
        mc_drawdowns: Optional[List[float]] = None,
        # Regression
        baseline_sharpe: float = 0.0,
        # Latency
        p99_latency_ms: float = 0.0,
    ) -> GateReport:
        """Run all quality checks and return a GateReport."""
        import time
        start = time.time()

        all_validations: List[ValidationResult] = []

        # Performance
        all_validations.extend(self._perf.validate(
            sharpe, win_rate, max_drawdown, profit_factor, num_trades
        ))

        # Stability (if data available)
        if fold_sharpes:
            all_validations.extend(self._stability.validate(fold_sharpes))

        # Stress (if data available)
        if mc_sharpes and mc_drawdowns:
            all_validations.extend(self._stress.validate(mc_sharpes, mc_drawdowns))

        # Regression (if baseline exists)
        if baseline_sharpe > 0:
            all_validations.extend(self._regression.validate(sharpe, baseline_sharpe))

        # Latency
        if p99_latency_ms > 0:
            all_validations.extend(self._latency.validate(p99_latency_ms))

        failures = [v.check_name for v in all_validations if not v.passed]
        status = GateStatus.PASSED if not failures else GateStatus.FAILED

        return GateReport(
            status=status,
            triggered_by=triggered_by,
            validations=all_validations,
            failures=failures,
            duration_seconds=time.time() - start,
        )
