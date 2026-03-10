"""
TITAN — Stress Validator
Checks Monte Carlo worst-case and stress test results.
"""

from __future__ import annotations

from typing import List

from aphelion.risk.titan.gate import ValidationResult


class StressValidator:
    """Validates stress test and Monte Carlo results."""

    def validate(
        self,
        mc_p5_sharpe: float,
        mc_p1_drawdown: float,
        stress_scenarios_passed: int,
        stress_scenarios_total: int,
        mc_min_p5_sharpe: float = 0.8,
        mc_max_p1_drawdown: float = 0.20,
        min_stress_pass_rate: float = 0.80,
    ) -> List[ValidationResult]:
        results = []

        results.append(ValidationResult(
            check_name="mc_p5_sharpe",
            passed=mc_p5_sharpe >= mc_min_p5_sharpe,
            actual_value=mc_p5_sharpe,
            threshold=mc_min_p5_sharpe,
            message="Monte Carlo 5th percentile Sharpe",
        ))
        results.append(ValidationResult(
            check_name="mc_p1_drawdown",
            passed=mc_p1_drawdown <= mc_max_p1_drawdown,
            actual_value=mc_p1_drawdown,
            threshold=mc_max_p1_drawdown,
            message="Monte Carlo 1st percentile drawdown",
        ))

        pass_rate = stress_scenarios_passed / stress_scenarios_total if stress_scenarios_total > 0 else 0.0
        results.append(ValidationResult(
            check_name="stress_pass_rate",
            passed=pass_rate >= min_stress_pass_rate,
            actual_value=pass_rate,
            threshold=min_stress_pass_rate,
            message=f"{stress_scenarios_passed}/{stress_scenarios_total} scenarios passed",
        ))
        return results
