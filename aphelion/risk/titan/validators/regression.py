"""
TITAN — Regression Validator
Ensures new changes don't degrade existing performance.
"""

from __future__ import annotations

from typing import Dict, List

from aphelion.risk.titan.gate import ValidationResult


class RegressionValidator:
    """Validates that new changes don't regress performance."""

    def validate(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        max_regression_pct: float = 0.10,
    ) -> List[ValidationResult]:
        results = []

        for metric, baseline_val in baseline_metrics.items():
            current_val = current_metrics.get(metric, 0.0)
            if baseline_val == 0:
                continue

            # For drawdown, regression means it got worse (higher)
            if "drawdown" in metric.lower():
                regressed = current_val > baseline_val * (1 + max_regression_pct)
                results.append(ValidationResult(
                    check_name=f"regression_{metric}",
                    passed=not regressed,
                    actual_value=current_val,
                    threshold=baseline_val * (1 + max_regression_pct),
                    message=f"Baseline: {baseline_val:.4f}",
                ))
            else:
                # For positive metrics (sharpe, win_rate), regression means lower
                regressed = current_val < baseline_val * (1 - max_regression_pct)
                results.append(ValidationResult(
                    check_name=f"regression_{metric}",
                    passed=not regressed,
                    actual_value=current_val,
                    threshold=baseline_val * (1 - max_regression_pct),
                    message=f"Baseline: {baseline_val:.4f}",
                ))

        return results
