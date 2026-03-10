"""
HEPHAESTUS — Agent

Autonomous orchestration loop that drives the entire forge pipeline.

State machine:
  IDLE → PARSING → GENERATING → SANDBOXING → FIXING (loop) →
  VALIDATING → DEPLOYING / REJECTING
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from aphelion.hephaestus.codegen import HephaestusCodegen
from aphelion.hephaestus.fixer import HephaestusFixer
from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import (
    ForgeResult,
    ForgeStatus,
    ForgedStrategy,
    RejectionReport,
    StrategySpec,
    ValidationReport,
)
from aphelion.hephaestus.parser import HephaestusParser
from aphelion.hephaestus.sandbox import HephaestusSandbox
from aphelion.hephaestus.validator import HephaestusValidator

logger = logging.getLogger(__name__)


class HephaestusAgent:
    """Autonomous forge agent — orchestrates the full pipeline.

    Call ``forge(source_code)`` with raw indicator source and receive a
    ``ForgeResult`` with status ``SHADOW`` / ``DEPLOYED`` or ``REJECTED``.
    """

    MAX_FIX_ATTEMPTS: int = 5
    MAX_PARSE_ATTEMPTS: int = 3
    SANDBOX_TIMEOUT_SECONDS: int = 30

    def __init__(
        self,
        llm_client: Optional[HephaestusLLMClient] = None,
        validator: Optional[HephaestusValidator] = None,
        deployer: Optional[object] = None,  # HephaestusDeployer
    ) -> None:
        self._llm = llm_client or HephaestusLLMClient()
        self._parser = HephaestusParser(self._llm)
        self._codegen = HephaestusCodegen(self._llm)
        self._fixer = HephaestusFixer(self._llm)
        self._sandbox = HephaestusSandbox(timeout=self.SANDBOX_TIMEOUT_SECONDS)
        self._validator = validator or HephaestusValidator()
        self._deployer = deployer

    # ── Public API ───────────────────────────────────────────────────────

    def forge(self, source_code: str) -> ForgeResult:
        """Full autonomous pipeline.

        Accepts raw source code and returns ``ForgeResult`` with final status.
        """
        result = ForgeResult(
            strategy_id=str(uuid.uuid4()),
            status=ForgeStatus.PENDING,
        )

        # Phase 1: Parse
        result.status = ForgeStatus.PARSING
        spec = self._parse_with_retry(source_code, result)
        if spec is None:
            return self._reject(result, ["Parser failed after max attempts"])
        result.spec = spec

        # Phase 2: Generate
        result.status = ForgeStatus.GENERATING
        forged = self._generate(spec, result)
        if forged is None:
            return self._reject(result, ["Code generation failed"])
        result.forged = forged

        # Phase 3: Sandbox + Fix loop
        result.status = ForgeStatus.TESTING
        forged = self._sandbox_and_fix_loop(forged, result)
        if forged is None:
            return self._reject(
                result,
                [f"Code failed after {self.MAX_FIX_ATTEMPTS} fix attempts"],
            )
        result.forged = forged

        # Phase 4: Validate
        result.status = ForgeStatus.VALIDATING
        validation = self._validate(forged, spec)
        result.validation = validation

        if not validation.passed:
            return self._reject(result, validation.rejection_reasons)

        # Phase 5: Deploy (if deployer is wired)
        if self._deployer is not None:
            try:
                voter_id = self._deployer.deploy(forged, spec, validation)  # type: ignore[attr-defined]
                result.ares_voter_id = voter_id
                result.status = ForgeStatus.SHADOW
            except Exception as exc:
                logger.error("Deployment failed: %s", exc)
                result.status = ForgeStatus.SHADOW
        else:
            result.status = ForgeStatus.SHADOW

        result.completed_at = datetime.now(timezone.utc)
        result.total_seconds = (
            result.completed_at - result.submitted_at
        ).total_seconds()
        result.total_llm_calls = self._llm.total_calls
        result.total_tokens_used = self._llm.total_tokens
        return result

    # ── Internal stages ──────────────────────────────────────────────────

    def _parse_with_retry(
        self, source_code: str, result: ForgeResult
    ) -> Optional[StrategySpec]:
        """Parse source code with retries."""
        for _ in range(self.MAX_PARSE_ATTEMPTS):
            result.parse_attempts += 1
            spec = self._parser.parse(source_code)
            if spec is not None:
                ok, issues = self._parser.validate_spec(spec)
                if ok:
                    return spec
                logger.info("Parse attempt %d: %s", result.parse_attempts, issues)
            else:
                logger.info("Parse attempt %d returned None", result.parse_attempts)
        return None

    def _generate(
        self, spec: StrategySpec, result: ForgeResult
    ) -> Optional[ForgedStrategy]:
        """Generate Python code from spec."""
        result.generation_attempts += 1
        return self._codegen.generate(spec)

    def _sandbox_and_fix_loop(
        self, forged: ForgedStrategy, result: ForgeResult
    ) -> Optional[ForgedStrategy]:
        """Run code in sandbox, fix errors up to MAX_FIX_ATTEMPTS."""
        for attempt in range(self.MAX_FIX_ATTEMPTS):
            sandbox_result = self._sandbox.execute(forged.python_code)

            if sandbox_result.success:
                # Also run unit tests
                test_result = self._sandbox.run_unit_tests(forged)
                if test_result.all_passed:
                    return forged
                error = f"Unit test failures:\n{test_result.failure_summary}"
            else:
                error = sandbox_result.error_message

            # Fix attempt
            result.fix_attempts += 1
            fixed_code = self._fixer.fix(forged, error)
            if fixed_code is None:
                continue

            forged.python_code = fixed_code
            forged.version += 1
            forged.fix_history.append(error)
            logger.info(
                "Fix attempt %d (version %d): %s",
                attempt + 1,
                forged.version,
                error[:100],
            )

        return None

    def _validate(
        self, forged: ForgedStrategy, spec: StrategySpec
    ) -> ValidationReport:
        """Run full validation pipeline."""
        return self._validator.validate(forged, spec)

    @staticmethod
    def _reject(result: ForgeResult, reasons: list[str]) -> ForgeResult:
        """Mark a result as REJECTED."""
        result.status = ForgeStatus.REJECTED
        result.completed_at = datetime.now(timezone.utc)
        result.total_seconds = (
            result.completed_at - result.submitted_at
        ).total_seconds()
        if result.validation is None:
            result.validation = ValidationReport()
        result.validation.rejection_reasons.extend(reasons)
        return result

    # ── Rejection report ─────────────────────────────────────────────────

    @staticmethod
    def build_rejection_report(result: ForgeResult, source_code: str) -> RejectionReport:
        """Build a detailed rejection report from a failed forge."""
        name = result.spec.name if result.spec else "Unknown"
        val = result.validation or ValidationReport()
        return RejectionReport(
            strategy_id=result.strategy_id,
            strategy_name=name,
            source_snippet=source_code[:200],
            failed_at=result.status.value,
            reasons=val.rejection_reasons,
            sharpe_ratio=val.sharpe_ratio,
            win_rate=val.win_rate,
            max_drawdown=val.max_drawdown,
            total_trades=val.total_trades,
            suggestions=val.recommendations,
        )
