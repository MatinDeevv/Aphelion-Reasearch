"""
TITAN — Performance Validator
Checks minimum Sharpe, win rate, profit factor, drawdown.
"""

from __future__ import annotations

from typing import List

from aphelion.risk.titan.gate import ValidationResult, TITAN_REQUIREMENTS


class PerformanceValidator:
    """Validates trading performance meets TITAN requirements."""

    def validate(
        self,
        sharpe: float,
        win_rate: float,
        max_drawdown: float,
        profit_factor: float,
        total_trades: int,
    ) -> List[ValidationResult]:
        results = []
        reqs = TITAN_REQUIREMENTS

        results.append(ValidationResult(
            check_name="sharpe_ratio",
            passed=sharpe >= reqs["min_sharpe_ratio"],
            actual_value=sharpe,
            threshold=reqs["min_sharpe_ratio"],
        ))
        results.append(ValidationResult(
            check_name="win_rate",
            passed=win_rate >= reqs["min_win_rate"],
            actual_value=win_rate,
            threshold=reqs["min_win_rate"],
        ))
        results.append(ValidationResult(
            check_name="max_drawdown",
            passed=max_drawdown <= reqs["max_drawdown"],
            actual_value=max_drawdown,
            threshold=reqs["max_drawdown"],
        ))
        results.append(ValidationResult(
            check_name="profit_factor",
            passed=profit_factor >= reqs["min_profit_factor"],
            actual_value=profit_factor,
            threshold=reqs["min_profit_factor"],
        ))
        results.append(ValidationResult(
            check_name="trade_count",
            passed=total_trades >= reqs["min_trades_for_significance"],
            actual_value=float(total_trades),
            threshold=float(reqs["min_trades_for_significance"]),
        ))
        return results
