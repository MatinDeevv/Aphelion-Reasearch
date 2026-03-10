"""
TITAN — Stability Validator
Checks walk-forward fold stability.
"""

from __future__ import annotations

from typing import List

import numpy as np

from aphelion.risk.titan.gate import ValidationResult, TITAN_REQUIREMENTS


class StabilityValidator:
    """Validates walk-forward stability metrics."""

    def validate(
        self,
        fold_sharpes: List[float],
        min_passing_folds: int = 0,
    ) -> List[ValidationResult]:
        reqs = TITAN_REQUIREMENTS
        results = []

        if not fold_sharpes:
            results.append(ValidationResult(
                check_name="wf_folds_available",
                passed=False, actual_value=0.0,
                threshold=float(reqs["wf_min_folds_passing"]),
                message="No walk-forward data",
            ))
            return results

        passing = sum(1 for s in fold_sharpes if s >= reqs["wf_min_median_sharpe"])
        median = float(np.median(fold_sharpes))
        variance = float(np.var(fold_sharpes))

        results.append(ValidationResult(
            check_name="wf_folds_passing",
            passed=passing >= reqs["wf_min_folds_passing"],
            actual_value=float(passing),
            threshold=float(reqs["wf_min_folds_passing"]),
        ))
        results.append(ValidationResult(
            check_name="wf_median_sharpe",
            passed=median >= reqs["wf_min_median_sharpe"],
            actual_value=median,
            threshold=reqs["wf_min_median_sharpe"],
        ))
        results.append(ValidationResult(
            check_name="wf_sharpe_variance",
            passed=variance <= reqs["wf_max_sharpe_variance"],
            actual_value=variance,
            threshold=reqs["wf_max_sharpe_variance"],
        ))
        return results
