"""
Tests for hephaestus.agent — autonomous forge pipeline.
"""

from __future__ import annotations

import pytest

from aphelion.hephaestus.agent import HephaestusAgent
from aphelion.hephaestus.llm_client import HephaestusLLMClient
from aphelion.hephaestus.models import (
    ForgeResult,
    ForgeStatus,
    ForgedStrategy,
    InputType,
    RejectionReport,
    StrategySpec,
    ValidationReport,
)
from aphelion.hephaestus.validator import HephaestusValidator


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _stub_agent() -> HephaestusAgent:
    """Agent with stub LLM (no API key → all LLM calls return empty)."""
    return HephaestusAgent(
        llm_client=HephaestusLLMClient(),
        validator=HephaestusValidator(),
    )


# ─── Full pipeline (stub) ───────────────────────────────────────────────────


class TestForgeStub:
    """In stub mode the LLM returns nothing → immediate rejection."""

    def test_forge_returns_rejected(self):
        agent = _stub_agent()
        result = agent.forge("some pine script")
        assert result.status == ForgeStatus.REJECTED
        assert result.strategy_id  # UUID assigned

    def test_forge_records_parse_attempts(self):
        agent = _stub_agent()
        result = agent.forge("anything")
        assert result.parse_attempts == agent.MAX_PARSE_ATTEMPTS

    def test_forge_sets_completed_at(self):
        agent = _stub_agent()
        result = agent.forge("x")
        assert result.completed_at is not None
        assert result.total_seconds >= 0

    def test_forge_rejection_has_reasons(self):
        agent = _stub_agent()
        result = agent.forge("test")
        assert result.validation is not None
        assert len(result.validation.rejection_reasons) > 0


# ─── _reject helper ─────────────────────────────────────────────────────────


class TestRejectHelper:

    def test_reject_sets_status(self):
        result = ForgeResult(strategy_id="test-1", status=ForgeStatus.PARSING)
        rejected = HephaestusAgent._reject(result, ["reason1", "reason2"])
        assert rejected.status == ForgeStatus.REJECTED
        assert "reason1" in rejected.validation.rejection_reasons
        assert "reason2" in rejected.validation.rejection_reasons

    def test_reject_preserves_existing_validation(self):
        result = ForgeResult(
            strategy_id="test-2",
            status=ForgeStatus.VALIDATING,
            validation=ValidationReport(sharpe_ratio=0.5),
        )
        rejected = HephaestusAgent._reject(result, ["new_reason"])
        assert rejected.validation.sharpe_ratio == 0.5
        assert "new_reason" in rejected.validation.rejection_reasons


# ─── build_rejection_report ──────────────────────────────────────────────────


class TestBuildRejectionReport:

    def test_builds_report(self):
        spec = StrategySpec(
            name="Test Strat",
            source_type=InputType.PINE_SCRIPT,
            description="test",
            confidence=0.9,
        )
        result = ForgeResult(
            strategy_id="abc-123",
            status=ForgeStatus.REJECTED,
            spec=spec,
            validation=ValidationReport(
                sharpe_ratio=0.8,
                win_rate=0.45,
                rejection_reasons=["Low Sharpe"],
                recommendations=["Add trend filter"],
            ),
        )
        report = HephaestusAgent.build_rejection_report(result, "some source code here")
        assert isinstance(report, RejectionReport)
        assert report.strategy_id == "abc-123"
        assert report.strategy_name == "Test Strat"
        assert report.source_snippet == "some source code here"
        assert "Low Sharpe" in report.reasons

    def test_report_handles_no_spec(self):
        result = ForgeResult(
            strategy_id="no-spec",
            status=ForgeStatus.REJECTED,
        )
        report = HephaestusAgent.build_rejection_report(result, "x")
        assert report.strategy_name == "Unknown"


# ─── Agent config ────────────────────────────────────────────────────────────


class TestAgentConfig:

    def test_default_max_fix_attempts(self):
        agent = _stub_agent()
        assert agent.MAX_FIX_ATTEMPTS == 5

    def test_default_max_parse_attempts(self):
        agent = _stub_agent()
        assert agent.MAX_PARSE_ATTEMPTS == 3

    def test_default_sandbox_timeout(self):
        agent = _stub_agent()
        assert agent.SANDBOX_TIMEOUT_SECONDS == 30
