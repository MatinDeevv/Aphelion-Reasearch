"""
Tests for hephaestus.deployer — deployment, shadow mode, registry.
"""

from __future__ import annotations

import pytest

from aphelion.hephaestus.deployer import (
    HephaestusDeployer,
    ShadowModeTracker,
    ShadowRecord,
)
from aphelion.hephaestus.models import (
    ForgedStrategy,
    ForgeResult,
    ForgeStatus,
    InputType,
    RejectionReport,
    ShadowEvaluation,
    StrategySpec,
    ValidationReport,
    Vote,
)
from aphelion.hephaestus.registry import HephaestusRegistry


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="Test Strategy",
        source_type=InputType.PINE_SCRIPT,
        description="test desc",
        confidence=0.9,
    )


def _make_forged() -> ForgedStrategy:
    return ForgedStrategy(
        spec=_make_spec(),
        python_code="class TestVoter: pass",
        class_name="TestVoter",
    )


def _make_validation(sharpe: float = 1.5) -> ValidationReport:
    return ValidationReport(sharpe_ratio=sharpe, passed=True)


# ─── ShadowModeTracker ──────────────────────────────────────────────────────


class TestShadowModeTracker:

    def test_register_and_count(self):
        t = ShadowModeTracker()
        t.register("v1", 1.5)
        assert t.trade_count("v1") == 0

    def test_record_increments_count(self):
        t = ShadowModeTracker()
        t.register("v1", 1.5)
        vote = Vote(1, 0.7, "BUY")
        t.record_shadow_vote("v1", vote, outcome=1, r_multiple=1.2)
        assert t.trade_count("v1") == 1

    def test_unregistered_voter_ignored(self):
        t = ShadowModeTracker()
        vote = Vote(1, 0.7, "BUY")
        t.record_shadow_vote("unknown", vote, outcome=1)
        assert t.trade_count("unknown") == 0

    def test_not_ready_before_threshold(self):
        t = ShadowModeTracker()
        t.SHADOW_TRADE_THRESHOLD = 100
        t.register("v1", 1.5)
        for i in range(50):
            t.record_shadow_vote("v1", Vote(1, 0.5, "B"), outcome=1, r_multiple=0.5)
        assert t.is_ready_for_evaluation("v1") is False

    def test_ready_after_threshold(self):
        t = ShadowModeTracker()
        t.SHADOW_TRADE_THRESHOLD = 10
        t.register("v1", 1.5)
        for i in range(10):
            t.record_shadow_vote("v1", Vote(1, 0.5, "B"), outcome=1, r_multiple=1.0)
        assert t.is_ready_for_evaluation("v1") is True

    def test_promote_good_performer(self):
        t = ShadowModeTracker()
        t.SHADOW_TRADE_THRESHOLD = 10
        t.register("v1", 1.0)
        # All positive R-multiples → good shadow Sharpe
        for _ in range(10):
            t.record_shadow_vote("v1", Vote(1, 0.5, "B"), outcome=1, r_multiple=2.0)
        result = t.evaluate_for_promotion("v1")
        assert result == ShadowEvaluation.PROMOTE

    def test_reject_poor_performer(self):
        t = ShadowModeTracker()
        t.SHADOW_TRADE_THRESHOLD = 10
        t.register("v1", 5.0)  # High validated Sharpe
        # All negative R-multiples → big regression
        for _ in range(10):
            t.record_shadow_vote("v1", Vote(1, 0.5, "B"), outcome=-1, r_multiple=-1.0)
        result = t.evaluate_for_promotion("v1")
        assert result == ShadowEvaluation.REJECT

    def test_continue_shadow_below_threshold(self):
        t = ShadowModeTracker()
        t.SHADOW_TRADE_THRESHOLD = 100
        t.register("v1", 1.0)
        result = t.evaluate_for_promotion("v1")
        assert result == ShadowEvaluation.CONTINUE_SHADOW


# ─── HephaestusDeployer ─────────────────────────────────────────────────────


class TestHephaestusDeployer:

    def test_deploy_returns_voter_id(self):
        d = HephaestusDeployer()
        vid = d.deploy(_make_forged(), _make_spec(), _make_validation())
        assert vid.startswith("heph_")

    def test_deploy_shadow_mode(self):
        d = HephaestusDeployer()
        vid = d.deploy(_make_forged(), _make_spec(), _make_validation(), mode="SHADOW")
        status = d.get_status(vid)
        assert status["mode"] == "SHADOW"

    def test_promote_voter(self):
        d = HephaestusDeployer()
        vid = d.deploy(_make_forged(), _make_spec(), _make_validation())
        assert d.promote(vid) is True
        assert d.get_status(vid)["mode"] == "FULL"

    def test_promote_unknown_voter(self):
        d = HephaestusDeployer()
        assert d.promote("nonexistent") is False

    def test_revoke_voter(self):
        d = HephaestusDeployer()
        vid = d.deploy(_make_forged(), _make_spec(), _make_validation())
        assert d.revoke(vid, "edge decayed") is True
        assert d.get_status(vid) is None

    def test_revoke_unknown(self):
        d = HephaestusDeployer()
        assert d.revoke("nonexistent") is False

    def test_list_deployed(self):
        d = HephaestusDeployer()
        d.deploy(_make_forged(), _make_spec(), _make_validation())
        d.deploy(_make_forged(), _make_spec(), _make_validation())
        assert len(d.list_deployed()) == 2


# ─── HephaestusRegistry ─────────────────────────────────────────────────────


class TestHephaestusRegistry:

    def test_register_and_get(self):
        r = HephaestusRegistry()
        result = ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW)
        r.register(result)
        assert r.get("s1") is result

    def test_get_missing_returns_none(self):
        r = HephaestusRegistry()
        assert r.get("missing") is None

    def test_list_by_status(self):
        r = HephaestusRegistry()
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.REJECTED))
        r.register(ForgeResult(strategy_id="s3", status=ForgeStatus.SHADOW))
        assert len(r.list_by_status(ForgeStatus.SHADOW)) == 2
        assert len(r.list_by_status(ForgeStatus.REJECTED)) == 1

    def test_list_deployed(self):
        r = HephaestusRegistry()
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.DEPLOYED))
        r.register(ForgeResult(strategy_id="s3", status=ForgeStatus.REJECTED))
        assert len(r.list_deployed()) == 2

    def test_total_counts(self):
        r = HephaestusRegistry()
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.REJECTED))
        assert r.total_forged == 2
        assert r.total_deployed == 1
        assert r.total_rejected == 1

    def test_success_rate(self):
        r = HephaestusRegistry()
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.REJECTED))
        assert r.get_success_rate() == 0.5

    def test_success_rate_empty(self):
        r = HephaestusRegistry()
        assert r.get_success_rate() == 0.0

    def test_success_rate_by_source_type(self):
        r = HephaestusRegistry()
        spec_pine = StrategySpec(name="A", source_type=InputType.PINE_SCRIPT, description="", confidence=0.9)
        spec_py = StrategySpec(name="B", source_type=InputType.PYTHON, description="", confidence=0.9)
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW, spec=spec_pine))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.REJECTED, spec=spec_pine))
        r.register(ForgeResult(strategy_id="s3", status=ForgeStatus.SHADOW, spec=spec_py))
        rates = r.get_success_rate_by_source_type()
        assert rates["PINE_SCRIPT"] == 0.5
        assert rates["PYTHON"] == 1.0

    def test_common_rejection_reasons(self):
        r = HephaestusRegistry()
        v1 = ValidationReport(rejection_reasons=["Low Sharpe", "Too few trades"])
        v2 = ValidationReport(rejection_reasons=["Low Sharpe"])
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.REJECTED, validation=v1))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.REJECTED, validation=v2))
        reasons = r.get_common_rejection_reasons()
        assert reasons[0] == ("Low Sharpe", 2)

    def test_search(self):
        r = HephaestusRegistry()
        spec = StrategySpec(name="EMA Crossover", source_type=InputType.PINE_SCRIPT, description="buy on cross", confidence=0.9)
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW, spec=spec))
        results = r.search("ema")
        assert len(results) == 1
        results2 = r.search("rsi")
        assert len(results2) == 0

    def test_best_performing_deployed(self):
        r = HephaestusRegistry()
        r.register(ForgeResult(strategy_id="s1", status=ForgeStatus.SHADOW, validation=ValidationReport(sharpe_ratio=1.5)))
        r.register(ForgeResult(strategy_id="s2", status=ForgeStatus.SHADOW, validation=ValidationReport(sharpe_ratio=2.0)))
        best = r.get_best_performing_deployed(top_n=1)
        assert len(best) == 1
        assert best[0].strategy_id == "s2"
