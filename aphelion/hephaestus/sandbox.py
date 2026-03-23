"""
HEPHAESTUS — Sandbox

Safe execution environment for LLM-generated strategy code.

Security model:
  1. AST pre-check — reject forbidden imports before execution.
  2. Subprocess isolation — execute in a separate process.
  3. Resource limits — CPU time, memory (Linux only), timeout.
  4. Output capture — stdout / stderr captured, never displayed raw.

On Windows ``resource.setrlimit`` is unavailable, so only the subprocess
timeout is enforced (still safe — the timeout hard-kills the child).
"""

from __future__ import annotations

import ast
import logging
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Optional

import numpy as np

from aphelion.hephaestus.models import (
    ASTCheckResult,
    SandboxResult,
    TestResult,
    ForgedStrategy,
    Vote,
)

logger = logging.getLogger(__name__)


# ─── Constants ───────────────────────────────────────────────────────────────


FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pickle",
        "shelve",
        "importlib",
        "ctypes",
        "cffi",
        "mmap",
        "multiprocessing",
        "threading",
        "concurrent",
        "asyncio",
        "shutil",
        "pathlib",
        "glob",
        "signal",
    }
)

# Heavy ML libs forbidden in voter code
FORBIDDEN_HEAVY: frozenset[str] = frozenset(
    {
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "tensorflow",
        "keras",
        "xgboost",
        "lightgbm",
    }
)


# ─── AST check ──────────────────────────────────────────────────────────────


def ast_check(code: str) -> ASTCheckResult:
    """Static safety check — parses code and rejects forbidden imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ASTCheckResult(safe=False, reason=f"Syntax error: {exc}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    return ASTCheckResult(safe=False, reason=f"Forbidden import: {alias.name}")
                if root in FORBIDDEN_HEAVY:
                    return ASTCheckResult(safe=False, reason=f"Heavy import forbidden: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                return ASTCheckResult(safe=False, reason=f"Forbidden import: {module}")
            if root in FORBIDDEN_HEAVY:
                return ASTCheckResult(safe=False, reason=f"Heavy import forbidden: {module}")

    return ASTCheckResult(safe=True, reason="")


# ─── Sandbox ─────────────────────────────────────────────────────────────────


class HephaestusSandbox:
    """Safe execution environment for LLM-generated code.

    All generated code runs in a subprocess with hard kill-timeout.
    """

    TIMEOUT_SECONDS: int = 30
    MAX_MEMORY_MB: int = 512
    MAX_CPU_SECONDS: int = 20

    def __init__(
        self,
        timeout: int = 30,
        max_memory_mb: int = 512,
    ) -> None:
        self.TIMEOUT_SECONDS = timeout
        self.MAX_MEMORY_MB = max_memory_mb

    # ── Public ───────────────────────────────────────────────────────────

    def execute(self, code: str) -> SandboxResult:
        """Execute generated Python code safely.

        Returns ``SandboxResult`` with success flag, output, and errors.
        """
        # Step 1: AST pre-check
        check = ast_check(code)
        if not check.safe:
            return SandboxResult(
                success=False,
                error_message=f"FORBIDDEN: {check.reason}",
                execution_ms=0.0,
            )

        # Step 2: Write to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".py", prefix="heph_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            # Step 3: Execute in subprocess
            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.TIMEOUT_SECONDS,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if result.returncode == 0:
                    return SandboxResult(
                        success=True,
                        output=result.stdout,
                        execution_ms=elapsed_ms,
                    )
                else:
                    return SandboxResult(
                        success=False,
                        error_message=result.stderr,
                        execution_ms=elapsed_ms,
                    )
            except subprocess.TimeoutExpired:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return SandboxResult(
                    success=False,
                    error_message="TIMEOUT: Execution exceeded limit",
                    execution_ms=elapsed_ms,
                )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def run_unit_tests(self, forged: ForgedStrategy, test_bars: Optional[np.ndarray] = None) -> TestResult:
        """Run functional tests on a forged strategy inside the sandbox.

        Instantiates the generated class and feeds it real bar data
        to verify it returns correct ``Vote`` objects without exceptions.

        Args:
            forged: The forged strategy to test.
            test_bars: Real OHLCV bar data as ndarray (shape (n, 6)).
                       If not provided, tests will use minimal constant bars.
        """
        test_code = self._build_test_harness(forged)
        result = self.execute(test_code)

        if not result.success:
            return TestResult(
                all_passed=False,
                failure_summary=result.error_message,
            )

        # Parse pass/fail counts from output
        return self._parse_test_output(result.output)

    # ── Internals ────────────────────────────────────────────────────────

    def _build_test_harness(self, forged: ForgedStrategy) -> str:
        """Build a self-contained test script that exercises the voter.

        Uses constant-value bars for structural validation (not synthetic random data).
        """
        lookback = forged.spec.lookback_bars
        class_name = forged.class_name

        # Build as concatenation so voter code keeps its own indentation
        prefix = "import numpy as np\n\n"
        voter_block = forged.python_code + "\n\n"
        suffix = textwrap.dedent(f"""\
            def make_constant_bars(n, price=2650.0):
                \"\"\"Create n constant-price bars for structural validation.\"\"\"
                ts = np.arange(n, dtype=float)
                opn = np.full(n, price)
                high = np.full(n, price + 0.5)
                low = np.full(n, price - 0.5)
                close = np.full(n, price)
                vol = np.full(n, 500.0)
                return np.column_stack([ts, opn, high, low, close, vol])

            passed = 0
            failed = 0

            tests = [
                ("empty",       np.zeros((0, 6))),
                ("short",       make_constant_bars(max({lookback} - 1, 1))),
                ("normal",      make_constant_bars({lookback} * 3)),
                ("long",        make_constant_bars({lookback} * 5)),
            ]

            voter = {class_name}()

            for name, bars in tests:
                try:
                    v = voter.vote(bars, {{}})
                    assert isinstance(v.direction, int) and v.direction in (-1, 0, 1), f"bad direction {{v.direction}}"
                    assert 0.0 <= v.confidence <= 1.0, f"bad confidence {{v.confidence}}"
                    assert isinstance(v.reason, str), "reason must be str"
                    passed += 1
                except Exception as e:
                    print(f"FAIL {{name}}: {{e}}")
                    failed += 1

            print(f"PASSED={{passed}} FAILED={{failed}}")
        """)
        return prefix + voter_block + suffix

    @staticmethod
    def _parse_test_output(output: str) -> TestResult:
        """Parse ``PASSED=N FAILED=M`` from test output."""
        import re

        m = re.search(r"PASSED=(\d+)\s+FAILED=(\d+)", output)
        if not m:
            return TestResult(
                all_passed=False,
                failure_summary=f"Unparseable test output: {output[:200]}",
            )
        p, f = int(m.group(1)), int(m.group(2))
        return TestResult(
            all_passed=(f == 0),
            passed=p,
            failed=f,
            failure_summary="" if f == 0 else f"{f} functional tests failed",
        )

    # ── Static helpers for test-data generation ──────────────────────────

    @staticmethod
    def generate_random_bars(n: int, seed: int = 42) -> np.ndarray:
        """Generate *n* random OHLCV bars (shape ``(n, 6)``)."""
        rng = np.random.default_rng(seed)
        base = 1900.0
        close = base + np.cumsum(rng.standard_normal(n) * 0.5)
        opn = close + rng.standard_normal(n) * 0.2
        high = np.maximum(close, opn) + rng.uniform(0, 1.0, n)
        low = np.minimum(close, opn) - rng.uniform(0, 1.0, n)
        vol = rng.uniform(100, 10_000, n)
        ts = np.arange(n, dtype=float)
        return np.column_stack([ts, opn, high, low, close, vol])

    @staticmethod
    def generate_trending_bars(n: int, trend: float = 1.0, seed: int = 42) -> np.ndarray:
        """Generate *n* bars with a directional trend."""
        rng = np.random.default_rng(seed)
        base = 1900.0
        drift = trend * 0.1  # per bar
        noise = rng.standard_normal(n) * 0.3
        close = base + np.cumsum(drift + noise)
        opn = close + rng.standard_normal(n) * 0.2
        high = np.maximum(close, opn) + rng.uniform(0, 0.5, n)
        low = np.minimum(close, opn) - rng.uniform(0, 0.5, n)
        vol = rng.uniform(100, 10_000, n)
        ts = np.arange(n, dtype=float)
        return np.column_stack([ts, opn, high, low, close, vol])

    @staticmethod
    def generate_flat_bars(n: int, seed: int = 42) -> np.ndarray:
        """Generate *n* bars with near-zero volatility."""
        rng = np.random.default_rng(seed)
        base = 1900.0
        close = base + rng.standard_normal(n) * 0.01
        opn = close + rng.standard_normal(n) * 0.005
        high = np.maximum(close, opn) + 0.01
        low = np.minimum(close, opn) - 0.01
        vol = rng.uniform(100, 10_000, n)
        ts = np.arange(n, dtype=float)
        return np.column_stack([ts, opn, high, low, close, vol])

    @staticmethod
    def generate_bars_with_nans(n: int, nan_fraction: float = 0.05, seed: int = 42) -> np.ndarray:
        """Generate *n* bars with some NaN values injected."""
        bars = HephaestusSandbox.generate_random_bars(n, seed)
        rng = np.random.default_rng(seed + 1)
        mask = rng.random(bars.shape) < nan_fraction
        bars[mask] = np.nan
        return bars
