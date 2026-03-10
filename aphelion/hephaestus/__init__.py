"""
HEPHAESTUS — Autonomous Strategy Factory

Accepts raw indicator code (Pine Script, Python, pseudocode, or plain English)
and autonomously parses, translates, validates, and deploys it as an
ARES-compatible strategy voter.

Quick start::

    from aphelion.hephaestus import Hephaestus

    forge = Hephaestus()
    result = forge.forge('''
        //@version=5
        indicator("My RSI Strategy")
        rsi = ta.rsi(close, 14)
        longCondition = ta.crossover(rsi, 30)
        shortCondition = ta.crossunder(rsi, 70)
    ''')
    print(result.status)
"""

from __future__ import annotations

from aphelion.hephaestus.models import (
    ASTCheckResult,
    CorrelationReport,
    ForgeResult,
    ForgeStatus,
    ForgedStrategy,
    InputType,
    RejectionReport,
    SandboxResult,
    ShadowEvaluation,
    StrategySpec,
    TestResult,
    ValidationReport,
    Vote,
    HEPHAESTUS_TITAN_REQUIREMENTS,
)
from aphelion.hephaestus.agent import HephaestusAgent
from aphelion.hephaestus.codegen import HephaestusCodegen
from aphelion.hephaestus.deployer import HephaestusDeployer, ShadowModeTracker
from aphelion.hephaestus.fixer import HephaestusFixer
from aphelion.hephaestus.llm_client import HephaestusLLMClient, LLMConfig
from aphelion.hephaestus.parser import HephaestusParser, ParserValidator, detect_input_type
from aphelion.hephaestus.registry import HephaestusRegistry
from aphelion.hephaestus.sandbox import HephaestusSandbox, ast_check
from aphelion.hephaestus.validator import (
    HephaestusValidator,
    titan_gate,
    validate_correlation,
    validate_functional,
    validate_syntax,
)

__all__ = [
    # Top-level facade
    "Hephaestus",
    # Models
    "ASTCheckResult",
    "CorrelationReport",
    "ForgeResult",
    "ForgeStatus",
    "ForgedStrategy",
    "InputType",
    "RejectionReport",
    "SandboxResult",
    "ShadowEvaluation",
    "StrategySpec",
    "TestResult",
    "ValidationReport",
    "Vote",
    "HEPHAESTUS_TITAN_REQUIREMENTS",
    # Core components
    "HephaestusAgent",
    "HephaestusCodegen",
    "HephaestusDeployer",
    "HephaestusFixer",
    "HephaestusLLMClient",
    "HephaestusParser",
    "HephaestusRegistry",
    "HephaestusSandbox",
    "HephaestusValidator",
    # Utilities
    "LLMConfig",
    "ParserValidator",
    "ShadowModeTracker",
    "ast_check",
    "detect_input_type",
    "titan_gate",
    "validate_correlation",
    "validate_functional",
    "validate_syntax",
]


class Hephaestus:
    """High-level facade for the HEPHAESTUS autonomous strategy factory.

    Provides a simple one-method API for the full forge pipeline plus
    convenience methods for querying status, listing deployments, and
    revoking strategies.
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        existing_voters: list[object] | None = None,
    ) -> None:
        self._llm = HephaestusLLMClient(llm_config)
        self._deployer = HephaestusDeployer()
        self._registry = HephaestusRegistry()
        self._validator = HephaestusValidator(existing_voters=existing_voters or [])
        self._agent = HephaestusAgent(
            llm_client=self._llm,
            validator=self._validator,
            deployer=self._deployer,
        )

    # ── Core ─────────────────────────────────────────────────────────────

    def forge(self, source_code: str) -> ForgeResult:
        """Full autonomous pipeline: parse → generate → test → validate → deploy."""
        result = self._agent.forge(source_code)
        self._registry.register(result)
        if result.status == ForgeStatus.REJECTED:
            report = HephaestusAgent.build_rejection_report(result, source_code)
            self._registry.register_rejection(report)
        return result

    # ── Query ────────────────────────────────────────────────────────────

    def get_status(self, strategy_id: str) -> ForgeResult | None:
        """Look up a forge result by strategy ID."""
        return self._registry.get(strategy_id)

    def list_deployed(self) -> list[ForgeResult]:
        """All deployed / shadowed strategies."""
        return self._registry.list_deployed()

    def list_rejected(self) -> list[ForgeResult]:
        """All rejected strategies."""
        return self._registry.list_rejected()

    # ── Management ───────────────────────────────────────────────────────

    def revoke(self, strategy_id: str, reason: str = "Edge decayed") -> bool:
        """Revoke a deployed strategy."""
        result = self._registry.get(strategy_id)
        if result is None or result.ares_voter_id is None:
            return False
        return self._deployer.revoke(result.ares_voter_id, reason)

    # ── Analytics ────────────────────────────────────────────────────────

    @property
    def registry(self) -> HephaestusRegistry:
        """Access the strategy registry for analytics."""
        return self._registry

    @property
    def deployer(self) -> HephaestusDeployer:
        """Access the deployer (shadow tracker, etc.)."""
        return self._deployer
