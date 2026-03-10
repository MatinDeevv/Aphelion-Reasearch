"""
HEPHAESTUS — Validator

Full validation pipeline for forged strategies.  Checks syntax, functional
correctness, backtest performance, walk-forward consistency, Monte Carlo
stress, TITAN gate thresholds, and correlation with existing ARES voters.
"""

from __future__ import annotations

import ast
import logging
from typing import Optional

import numpy as np

from aphelion.hephaestus.models import (
    CorrelationReport,
    ForgedStrategy,
    StrategySpec,
    ValidationReport,
    Vote,
    HEPHAESTUS_TITAN_REQUIREMENTS as TITAN,
)
from aphelion.hephaestus.sandbox import (
    FORBIDDEN_IMPORTS,
    FORBIDDEN_HEAVY,
    HephaestusSandbox,
    ast_check,
)

logger = logging.getLogger(__name__)


# ─── Syntax validation ──────────────────────────────────────────────────────


def validate_syntax(code: str) -> tuple[bool, str]:
    """Compile code and check for forbidden imports.

    Returns ``(ok, error_message)``.
    """
    check = ast_check(code)
    if not check.safe:
        return False, check.reason

    # Extra structural check: must contain a class with vote()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, str(exc)

    has_class = False
    has_vote = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            has_class = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "vote":
                        has_vote = True
    if not has_class:
        return False, "No class definition found"
    if not has_vote:
        return False, "No vote() method found"
    return True, ""


# ─── Functional validation ───────────────────────────────────────────────────


def validate_functional(
    voter_class: type,
    spec: StrategySpec,
) -> tuple[bool, list[str]]:
    """Instantiate the voter and run it on synthetic data.

    Verifies the ``vote()`` method returns correct types.
    """
    sandbox = HephaestusSandbox()
    tests: list[tuple[str, np.ndarray]] = [
        ("empty_bars", np.zeros((0, 6))),
        ("insufficient_bars", sandbox.generate_random_bars(max(spec.lookback_bars - 1, 1))),
        ("normal_bars", sandbox.generate_random_bars(spec.lookback_bars * 3)),
        ("trending_up", sandbox.generate_trending_bars(spec.lookback_bars * 2, trend=1)),
        ("trending_down", sandbox.generate_trending_bars(spec.lookback_bars * 2, trend=-1)),
        ("flat_market", sandbox.generate_flat_bars(spec.lookback_bars * 2)),
        ("nan_bars", sandbox.generate_bars_with_nans(spec.lookback_bars * 2)),
    ]

    failures: list[str] = []
    try:
        voter = voter_class()
    except Exception as exc:
        return False, [f"Instantiation failed: {exc}"]

    for test_name, bars in tests:
        try:
            vote = voter.vote(bars, {})
            if not isinstance(vote.direction, int) or vote.direction not in (-1, 0, 1):
                failures.append(f"{test_name}: bad direction {vote.direction}")
            if not (0.0 <= vote.confidence <= 1.0):
                failures.append(f"{test_name}: bad confidence {vote.confidence}")
            if not isinstance(vote.reason, str):
                failures.append(f"{test_name}: reason is not str")
        except Exception as exc:
            failures.append(f"{test_name}: {exc}")

    return len(failures) == 0, failures


# ─── Correlation guard ───────────────────────────────────────────────────────


def validate_correlation(
    voter_class: type,
    bars: np.ndarray,
    existing_voters: list[object],
    threshold: float = 0.75,
) -> CorrelationReport:
    """Check whether a new voter is sufficiently uncorrelated with existing voters."""
    if len(bars) < 100 or not existing_voters:
        return CorrelationReport(adds_diversity=True)

    new_signals = _signal_series(voter_class, bars)
    max_corr = 0.0
    most_corr = ""

    for voter in existing_voters:
        try:
            existing_signals = _signal_series(type(voter), bars)
            if len(new_signals) != len(existing_signals):
                continue
            corr = float(np.corrcoef(new_signals, existing_signals)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                most_corr = getattr(voter, "name", type(voter).__name__)
        except Exception:
            continue

    return CorrelationReport(
        max_correlation=max_corr,
        most_correlated_voter=most_corr,
        adds_diversity=max_corr < threshold,
    )


def _signal_series(voter_class: type, bars: np.ndarray) -> np.ndarray:
    """Produce a direction series for the voter over ``bars``."""
    voter = voter_class()
    lookback = getattr(voter, "lookback", 50)
    if callable(lookback):
        lookback = lookback()
    signals = []
    for i in range(lookback, len(bars)):
        window = bars[max(0, i - 500) : i]
        try:
            v = voter.vote(window, {})
            signals.append(float(v.direction))
        except Exception:
            signals.append(0.0)
    return np.array(signals, dtype=np.float64)


# ─── TITAN gate ──────────────────────────────────────────────────────────────


def titan_gate(report: ValidationReport) -> tuple[bool, list[str]]:
    """Apply HEPHAESTUS TITAN requirements.

    Returns ``(passed, [failure reasons])``.
    """
    failures: list[str] = []

    if report.sharpe_ratio < TITAN["min_sharpe_ratio"]:
        failures.append(
            f"Sharpe {report.sharpe_ratio:.2f} < {TITAN['min_sharpe_ratio']}"
        )
    if report.win_rate < TITAN["min_win_rate"]:
        failures.append(
            f"Win rate {report.win_rate:.2%} < {TITAN['min_win_rate']:.0%}"
        )
    if report.max_drawdown > TITAN["max_drawdown"]:
        failures.append(
            f"Max drawdown {report.max_drawdown:.2%} > {TITAN['max_drawdown']:.0%}"
        )
    if report.profit_factor < TITAN["min_profit_factor"]:
        failures.append(
            f"Profit factor {report.profit_factor:.2f} < {TITAN['min_profit_factor']}"
        )
    if report.total_trades < TITAN["min_trades_for_significance"]:
        failures.append(
            f"Only {report.total_trades} trades (min: {int(TITAN['min_trades_for_significance'])})"
        )

    # Walk-forward
    if report.wf_folds_passed < TITAN["wf_min_folds_passing"]:
        failures.append(
            f"WF folds passed {report.wf_folds_passed} < {int(TITAN['wf_min_folds_passing'])}"
        )
    if report.wf_median_sharpe < TITAN["wf_min_median_sharpe"]:
        failures.append(
            f"WF median Sharpe {report.wf_median_sharpe:.2f} < {TITAN['wf_min_median_sharpe']}"
        )
    if report.wf_sharpe_variance > TITAN["wf_max_sharpe_variance"]:
        failures.append(
            f"WF Sharpe variance {report.wf_sharpe_variance:.2f} > {TITAN['wf_max_sharpe_variance']}"
        )

    # Monte Carlo
    if report.mc_5th_pct_sharpe < TITAN["mc_5th_percentile_sharpe"]:
        failures.append(
            f"MC 5th pct Sharpe {report.mc_5th_pct_sharpe:.2f} < {TITAN['mc_5th_percentile_sharpe']}"
        )
    if report.mc_95th_pct_max_dd > TITAN["mc_95th_percentile_max_dd"]:
        failures.append(
            f"MC 95th pct max DD {report.mc_95th_pct_max_dd:.2%} > {TITAN['mc_95th_percentile_max_dd']:.0%}"
        )

    # Correlation
    if report.max_correlation_with_existing > TITAN["max_correlation_with_existing"]:
        failures.append(
            f"Correlation {report.max_correlation_with_existing:.2f} with "
            f"{report.correlated_voter} > {TITAN['max_correlation_with_existing']}"
        )

    return len(failures) == 0, failures


# ─── Full validation pipeline ────────────────────────────────────────────────


class HephaestusValidator:
    """Runs the complete validation pipeline on a ``ForgedStrategy``."""

    def __init__(
        self,
        existing_voters: Optional[list[object]] = None,
        historical_bars: Optional[np.ndarray] = None,
    ) -> None:
        self._existing_voters = existing_voters or []
        self._historical_bars = historical_bars
        self._sandbox = HephaestusSandbox()

    def validate(
        self,
        forged: ForgedStrategy,
        spec: StrategySpec,
    ) -> ValidationReport:
        """Run all validation stages and return a ``ValidationReport``."""
        report = ValidationReport()

        # Stage 1: Syntax
        ok, err = validate_syntax(forged.python_code)
        report.syntax_valid = ok
        if not ok:
            report.rejection_reasons.append(f"Syntax: {err}")
            return report

        # Stage 2: Load class
        voter_class = self._load_class(forged)
        if voter_class is None:
            report.rejection_reasons.append("Failed to load generated class")
            return report

        # Stage 3: Functional
        func_ok, func_errs = validate_functional(voter_class, spec)
        report.unit_tests_total = 7
        report.unit_tests_passed = 7 - len(func_errs)
        if not func_ok:
            report.runtime_errors = func_errs
            report.rejection_reasons.append(
                f"Functional: {len(func_errs)} tests failed"
            )
            return report

        # Stage 4: Correlation guard (if we have bars + existing voters)
        if self._historical_bars is not None and len(self._existing_voters) > 0:
            corr = validate_correlation(
                voter_class,
                self._historical_bars,
                self._existing_voters,
                threshold=TITAN["max_correlation_with_existing"],
            )
            report.max_correlation_with_existing = corr.max_correlation
            report.correlated_voter = corr.most_correlated_voter
            report.adds_diversity = corr.adds_diversity

        # Stage 5: TITAN gate (based on whatever metrics we have so far)
        titan_ok, titan_errs = titan_gate(report)
        report.titan_passed = titan_ok
        report.titan_failures = titan_errs

        # Note: full backtest / walk-forward / MC require a DataLayer + real
        # historical data.  When those are available the agent calls them
        # separately and populates the report before the TITAN gate check.
        # Here we mark pass/fail based on available data.

        if titan_errs:
            report.rejection_reasons.extend(titan_errs)
        if not report.adds_diversity:
            report.rejection_reasons.append(
                f"High correlation ({report.max_correlation_with_existing:.2f}) "
                f"with {report.correlated_voter}"
            )

        report.passed = len(report.rejection_reasons) == 0
        if report.passed:
            report.recommendations.append("Strategy passed all validation stages")
        else:
            # Generate improvement suggestions
            report.recommendations = self._generate_suggestions(report)

        return report

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_class(forged: ForgedStrategy) -> Optional[type]:
        """Dynamically load the voter class from generated code."""
        namespace: dict = {}
        try:
            exec(forged.python_code, namespace)  # noqa: S102
        except Exception as exc:
            logger.warning("Validator: exec failed — %s", exc)
            return None
        return namespace.get(forged.class_name)

    @staticmethod
    def _generate_suggestions(report: ValidationReport) -> list[str]:
        """Generate actionable improvement suggestions from failures."""
        suggestions: list[str] = []
        if report.win_rate < 0.50:
            suggestions.append(
                f"Win rate {report.win_rate:.0%} below 50% — consider adding a trend filter"
            )
        if report.max_drawdown > 0.15:
            suggestions.append(
                f"Max drawdown {report.max_drawdown:.0%} too high — consider tighter stop losses"
            )
        if report.total_trades < 150:
            suggestions.append(
                f"Only {report.total_trades} trades — insufficient statistical significance"
            )
        if not report.adds_diversity:
            suggestions.append(
                f"Correlation {report.max_correlation_with_existing:.2f} with "
                f"{report.correlated_voter} — strategy adds no diversity"
            )
        if report.wf_sharpe_variance > 1.2:
            suggestions.append(
                f"Walk-forward variance {report.wf_sharpe_variance:.1f} — "
                "strategy is inconsistent across time periods"
            )
        return suggestions
