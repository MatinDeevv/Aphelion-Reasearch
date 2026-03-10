"""
Tests for hephaestus.validator — syntax, functional, correlation, TITAN gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.hephaestus.models import (
    CorrelationReport,
    ForgedStrategy,
    InputType,
    StrategySpec,
    ValidationReport,
    Vote,
    HEPHAESTUS_TITAN_REQUIREMENTS,
)
from aphelion.hephaestus.validator import (
    HephaestusValidator,
    titan_gate,
    validate_correlation,
    validate_functional,
    validate_syntax,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


VALID_VOTER_CODE = '''\
import numpy as np
from abc import ABC, abstractmethod

class Vote:
    def __init__(self, direction, confidence, reason, metadata=None):
        self.direction = direction
        self.confidence = confidence
        self.reason = reason
        self.metadata = metadata or {}

class BaseARESVoter(ABC):
    tier = "COMMANDER"
    weight = 10
    @abstractmethod
    def vote(self, bars, context): ...
    @property
    @abstractmethod
    def lookback(self): ...
    @property
    @abstractmethod
    def name(self): ...

class ValidVoter(BaseARESVoter):
    @property
    def lookback(self):
        return 10
    @property
    def name(self):
        return "VALID_VOTER"
    def vote(self, bars, context):
        if len(bars) < self.lookback:
            return Vote(0, 0.0, "INSUFFICIENT_DATA")
        close = bars[:, 4].astype(float)
        if np.any(np.isnan(close)):
            return Vote(0, 0.0, "NAN_IN_DATA")
        return Vote(1, 0.6, "SIGNAL", {"close": float(close[-1])})
'''

BROKEN_VOTER_CODE = '''\
class BrokenVoter:
    def vote(self, bars, context):
        raise RuntimeError("always fails")
'''


def _make_spec(**overrides) -> StrategySpec:
    defaults = dict(
        name="Test",
        source_type=InputType.PYTHON,
        description="test",
        entry_long_conditions=["cond1"],
        lookback_bars=10,
        confidence=0.9,
    )
    defaults.update(overrides)
    return StrategySpec(**defaults)


def _make_forged(code: str = VALID_VOTER_CODE, class_name: str = "ValidVoter") -> ForgedStrategy:
    return ForgedStrategy(spec=_make_spec(), python_code=code, class_name=class_name)


def _load_class(code: str, name: str) -> type:
    ns = {}
    exec(code, ns)
    return ns[name]


# ─── validate_syntax ────────────────────────────────────────────────────────


class TestValidateSyntax:

    def test_valid_code_passes(self):
        ok, err = validate_syntax(VALID_VOTER_CODE)
        assert ok is True
        assert err == ""

    def test_syntax_error_fails(self):
        ok, err = validate_syntax("class Foo(\n  pass")
        assert ok is False
        assert "Syntax" in err or "invalid" in err.lower()

    def test_forbidden_import_fails(self):
        code = "import os\nclass X:\n    def vote(self): pass"
        ok, err = validate_syntax(code)
        assert ok is False

    def test_no_class_fails(self):
        code = "def vote(bars, ctx): pass"
        ok, err = validate_syntax(code)
        assert ok is False
        assert "class" in err.lower()

    def test_no_vote_method_fails(self):
        code = "class Foo:\n    def compute(self): pass"
        ok, err = validate_syntax(code)
        assert ok is False
        assert "vote" in err.lower()

    def test_real_class_with_vote_passes(self):
        code = "class MyVoter:\n    def vote(self, bars, ctx):\n        return None"
        ok, _ = validate_syntax(code)
        assert ok is True


# ─── validate_functional ────────────────────────────────────────────────────


class TestValidateFunctional:

    def test_valid_voter_passes(self):
        cls = _load_class(VALID_VOTER_CODE, "ValidVoter")
        ok, errs = validate_functional(cls, _make_spec())
        assert ok is True
        assert errs == []

    def test_broken_voter_fails(self):
        code = BROKEN_VOTER_CODE
        cls = _load_class(code, "BrokenVoter")
        ok, errs = validate_functional(cls, _make_spec())
        assert ok is False
        assert len(errs) > 0

    def test_bad_direction_detected(self):
        code = '''
class Vote:
    def __init__(self, direction, confidence, reason, metadata=None):
        self.direction = direction
        self.confidence = confidence
        self.reason = reason
        self.metadata = metadata or {}

class BadDirVoter:
    @property
    def lookback(self):
        return 5
    def vote(self, bars, ctx):
        return Vote(99, 0.5, "BAD")
'''
        cls = _load_class(code, "BadDirVoter")
        ok, errs = validate_functional(cls, _make_spec(lookback_bars=5))
        assert ok is False
        assert any("direction" in e.lower() for e in errs)


# ─── titan_gate ──────────────────────────────────────────────────────────────


class TestTITANGate:

    def test_perfect_report_passes(self):
        report = ValidationReport(
            sharpe_ratio=2.0,
            win_rate=0.60,
            max_drawdown=0.05,
            profit_factor=2.0,
            total_trades=300,
            wf_folds_passed=10,
            wf_median_sharpe=1.5,
            wf_sharpe_variance=0.5,
            mc_5th_pct_sharpe=1.0,
            mc_95th_pct_max_dd=0.15,
            max_correlation_with_existing=0.30,
        )
        ok, failures = titan_gate(report)
        assert ok is True
        assert failures == []

    def test_low_sharpe_fails(self):
        report = ValidationReport(sharpe_ratio=0.5)
        ok, failures = titan_gate(report)
        assert ok is False
        assert any("Sharpe" in f for f in failures)

    def test_low_win_rate_fails(self):
        report = ValidationReport(sharpe_ratio=2.0, win_rate=0.30)
        ok, failures = titan_gate(report)
        assert ok is False
        assert any("Win rate" in f for f in failures)

    def test_high_drawdown_fails(self):
        report = ValidationReport(sharpe_ratio=2.0, win_rate=0.6, max_drawdown=0.25)
        ok, failures = titan_gate(report)
        assert ok is False
        assert any("drawdown" in f.lower() for f in failures)

    def test_too_few_trades_fails(self):
        report = ValidationReport(
            sharpe_ratio=2.0, win_rate=0.6, max_drawdown=0.05,
            profit_factor=2.0, total_trades=50,
        )
        ok, failures = titan_gate(report)
        assert ok is False
        assert any("trades" in f.lower() for f in failures)

    def test_high_correlation_fails(self):
        report = ValidationReport(
            sharpe_ratio=2.0, win_rate=0.6, max_drawdown=0.05,
            profit_factor=2.0, total_trades=300,
            wf_folds_passed=10, wf_median_sharpe=1.5, wf_sharpe_variance=0.5,
            mc_5th_pct_sharpe=1.0, mc_95th_pct_max_dd=0.15,
            max_correlation_with_existing=0.90,
            correlated_voter="EXISTING_VOTER",
        )
        ok, failures = titan_gate(report)
        assert ok is False
        assert any("Correlation" in f for f in failures)


# ─── HephaestusValidator integration ─────────────────────────────────────────


class TestHephaestusValidator:

    def test_valid_forged_passes_syntax_and_functional(self):
        validator = HephaestusValidator()
        forged = _make_forged(VALID_VOTER_CODE, "ValidVoter")
        report = validator.validate(forged, _make_spec())
        assert report.syntax_valid is True
        assert report.unit_tests_passed == 7

    def test_syntax_failure_short_circuits(self):
        validator = HephaestusValidator()
        forged = _make_forged("import os\nclass X:\n    def vote(self): pass", "X")
        report = validator.validate(forged, _make_spec())
        assert report.syntax_valid is False
        assert len(report.rejection_reasons) > 0

    def test_generate_suggestions(self):
        report = ValidationReport(win_rate=0.30, max_drawdown=0.20, total_trades=50)
        suggestions = HephaestusValidator._generate_suggestions(report)
        assert any("Win rate" in s for s in suggestions)
        assert any("drawdown" in s.lower() for s in suggestions)
        assert any("trades" in s.lower() for s in suggestions)
