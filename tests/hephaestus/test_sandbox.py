"""
Tests for hephaestus.sandbox — AST checks, subprocess execution, security.
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.hephaestus.models import ForgedStrategy, InputType, StrategySpec
from aphelion.hephaestus.sandbox import (
    HephaestusSandbox,
    ast_check,
    FORBIDDEN_IMPORTS,
    FORBIDDEN_HEAVY,
)


# ─── AST checks ─────────────────────────────────────────────────────────────


class TestASTCheck:

    def test_clean_code_passes(self):
        code = "import numpy as np\nx = np.array([1, 2, 3])"
        result = ast_check(code)
        assert result.safe is True

    def test_forbidden_os_blocked(self):
        code = "import os\nos.listdir('.')"
        result = ast_check(code)
        assert result.safe is False
        assert "os" in result.reason

    def test_forbidden_subprocess_blocked(self):
        code = "import subprocess"
        result = ast_check(code)
        assert result.safe is False

    def test_forbidden_socket_blocked(self):
        code = "import socket"
        result = ast_check(code)
        assert result.safe is False

    def test_forbidden_sys_blocked(self):
        code = "import sys\nsys.exit(0)"
        result = ast_check(code)
        assert result.safe is False

    def test_forbidden_from_import_blocked(self):
        code = "from os.path import join"
        result = ast_check(code)
        assert result.safe is False

    def test_heavy_pandas_blocked(self):
        code = "import pandas as pd"
        result = ast_check(code)
        assert result.safe is False
        assert "Heavy" in result.reason or "pandas" in result.reason

    def test_heavy_torch_blocked(self):
        code = "import torch"
        result = ast_check(code)
        assert result.safe is False

    def test_syntax_error_caught(self):
        code = "def foo(\n  # incomplete"
        result = ast_check(code)
        assert result.safe is False
        assert "Syntax" in result.reason

    def test_numpy_allowed(self):
        code = "import numpy as np\nimport math"
        result = ast_check(code)
        assert result.safe is True

    def test_all_forbidden_imports_covered(self):
        """Ensure every forbidden import is actually blocked."""
        for mod in FORBIDDEN_IMPORTS:
            code = f"import {mod}"
            result = ast_check(code)
            assert result.safe is False, f"{mod} should be blocked"

    def test_all_heavy_imports_covered(self):
        for mod in FORBIDDEN_HEAVY:
            code = f"import {mod}"
            result = ast_check(code)
            assert result.safe is False, f"{mod} should be blocked"


# ─── Sandbox execution ──────────────────────────────────────────────────────


class TestSandboxExecution:

    def test_simple_print(self):
        sandbox = HephaestusSandbox(timeout=10)
        result = sandbox.execute('print("hello")')
        assert result.success is True
        assert "hello" in result.output

    def test_syntax_error_fails(self):
        sandbox = HephaestusSandbox(timeout=10)
        result = sandbox.execute("def foo(:\n  pass")
        assert result.success is False

    def test_forbidden_import_blocked(self):
        sandbox = HephaestusSandbox(timeout=10)
        result = sandbox.execute("import os; os.listdir('.')")
        assert result.success is False
        assert "FORBIDDEN" in result.error_message

    def test_runtime_error_reported(self):
        sandbox = HephaestusSandbox(timeout=10)
        result = sandbox.execute("x = 1 / 0")
        assert result.success is False
        assert "ZeroDivision" in result.error_message

    def test_execution_time_recorded(self):
        sandbox = HephaestusSandbox(timeout=10)
        result = sandbox.execute('print("fast")')
        assert result.execution_ms >= 0


# ─── Sandbox data generators ────────────────────────────────────────────────


class TestDataGenerators:

    def test_random_bars_shape(self):
        bars = HephaestusSandbox.generate_random_bars(100)
        assert bars.shape == (100, 6)

    def test_random_bars_no_nan(self):
        bars = HephaestusSandbox.generate_random_bars(50)
        assert not np.any(np.isnan(bars))

    def test_trending_bars_direction(self):
        up = HephaestusSandbox.generate_trending_bars(200, trend=1.0)
        down = HephaestusSandbox.generate_trending_bars(200, trend=-1.0)
        # Uptrend should end higher than start
        assert up[-1, 4] > up[0, 4]
        # Downtrend should end lower than start
        assert down[-1, 4] < down[0, 4]

    def test_flat_bars_low_vol(self):
        bars = HephaestusSandbox.generate_flat_bars(100)
        close = bars[:, 4]
        assert np.std(close) < 0.1  # Near-zero volatility

    def test_nan_bars_has_nans(self):
        bars = HephaestusSandbox.generate_bars_with_nans(200, nan_fraction=0.1)
        assert np.any(np.isnan(bars))


# ─── run_unit_tests ──────────────────────────────────────────────────────────


class TestRunUnitTests:

    def _make_forged(self, code: str, class_name: str = "TestVoter") -> ForgedStrategy:
        spec = StrategySpec(
            name="Test",
            source_type=InputType.PYTHON,
            description="test",
            lookback_bars=10,
            confidence=0.9,
        )
        return ForgedStrategy(spec=spec, python_code=code, class_name=class_name)

    def test_valid_voter_passes_tests(self):
        code = '''
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

class TestVoter(BaseARESVoter):
    @property
    def lookback(self):
        return 10
    @property
    def name(self):
        return "TEST"
    def vote(self, bars, context):
        if len(bars) < self.lookback:
            return Vote(0, 0.0, "INSUFFICIENT_DATA")
        return Vote(1, 0.5, "SIGNAL")
'''
        sandbox = HephaestusSandbox(timeout=15)
        result = sandbox.run_unit_tests(self._make_forged(code))
        assert result.all_passed is True
        assert result.passed == 4
        assert result.failed == 0
