"""
Phase 10 Tests — ARES Integration: Coordinator, Reasoner, Journal

Covers:
  - AresCoordinator: vote aggregation, consensus, regime filter, SENTINEL veto
  - AresReasoner: mock LLM, conflict resolution, thesis generation, fallback
  - DecisionJournal: record, outcome tracking, persistence, accuracy
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aphelion.core.config import Tier
from aphelion.ares.coordinator import (
    AresConfig, AresCoordinator, SignalSource, StrategyVote, AggregatedSignal,
)
from aphelion.ares.reasoner import (
    AresReasoner, ReasonerConfig, LLMProvider, ReasoningResult,
)
from aphelion.ares.journal import DecisionJournal, JournalEntry


# ═══════════════════════════════════════════════════════════════════════════
# Coordinator
# ═══════════════════════════════════════════════════════════════════════════

class TestAresCoordinator:
    def _make_coord(self, **kw) -> AresCoordinator:
        cfg = AresConfig(**kw)
        coord = AresCoordinator(cfg)
        coord._bars_since_decision = 999  # no cooldown
        return coord

    def _long_vote(self, source=SignalSource.HYDRA, conf=0.8, tier=Tier.MINISTER):
        return StrategyVote(source=source, direction=1, confidence=conf, tier=tier)

    def _short_vote(self, source=SignalSource.PROMETHEUS, conf=0.7, tier=Tier.MINISTER):
        return StrategyVote(source=source, direction=-1, confidence=conf, tier=tier)

    def test_unanimous_long(self):
        coord = self._make_coord(min_consensus_score=0.1, min_agreement_ratio=0.3, min_confidence=0.1)
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.9),
            self._long_vote(SignalSource.PROMETHEUS, 0.8),
        ]
        signal = coord.aggregate(votes)
        assert signal.direction == 1
        assert signal.consensus_score > 0
        assert signal.agreement_ratio > 0.5

    def test_unanimous_short(self):
        coord = self._make_coord(min_consensus_score=0.1, min_agreement_ratio=0.3, min_confidence=0.1)
        votes = [
            self._short_vote(SignalSource.HYDRA, 0.9),
            self._short_vote(SignalSource.PROMETHEUS, 0.85),
        ]
        signal = coord.aggregate(votes)
        assert signal.direction == -1

    def test_no_votes_returns_flat(self):
        coord = self._make_coord()
        signal = coord.aggregate([])
        assert signal.direction == 0

    def test_inactive_sources_filtered(self):
        coord = self._make_coord()
        coord.deactivate_source(SignalSource.HYDRA)
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.95),
        ]
        signal = coord.aggregate(votes)
        assert signal.direction == 0
        assert "No active" in signal.reasoning

    def test_sentinel_veto(self):
        coord = self._make_coord(min_consensus_score=0.1, min_agreement_ratio=0.3, min_confidence=0.1)
        coord.set_sentinel_veto(True)
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.9),
            self._long_vote(SignalSource.PROMETHEUS, 0.9),
        ]
        signal = coord.aggregate(votes)
        assert signal.vetoed

    def test_low_agreement_goes_flat(self):
        coord = self._make_coord(min_agreement_ratio=0.90)
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.6),
            self._short_vote(SignalSource.PROMETHEUS, 0.5),
        ]
        signal = coord.aggregate(votes)
        assert signal.direction == 0

    def test_regime_filter_reduces_confidence(self):
        coord = self._make_coord(
            use_regime_filter=True,
            min_consensus_score=0.1,
            min_agreement_ratio=0.3,
            min_confidence=0.01,
        )
        coord.set_regime("VOL_EXPANSION")
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.8),
            self._long_vote(SignalSource.PROMETHEUS, 0.7),
        ]
        signal = coord.aggregate(votes)
        assert "Vol-expansion" in signal.reasoning

    def test_cooldown_blocks_rapid_decisions(self):
        coord = self._make_coord(
            decision_cooldown_bars=5,
            min_consensus_score=0.1,
            min_agreement_ratio=0.3,
            min_confidence=0.1,
        )
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.9),
            self._long_vote(SignalSource.PROMETHEUS, 0.8),
        ]
        signal1 = coord.aggregate(votes)
        assert signal1.direction == 1
        # Second call immediately — should be blocked
        signal2 = coord.aggregate(votes)
        assert signal2.direction == 0
        assert "Cooldown" in signal2.reasoning

    def test_source_activation_deactivation(self):
        coord = self._make_coord()
        assert SignalSource.HYDRA in coord.active_sources
        coord.deactivate_source(SignalSource.HYDRA)
        assert SignalSource.HYDRA not in coord.active_sources
        coord.activate_source(SignalSource.HYDRA)
        assert SignalSource.HYDRA in coord.active_sources

    def test_decision_count_increments(self):
        coord = self._make_coord(
            min_consensus_score=0.1, min_agreement_ratio=0.3, min_confidence=0.1
        )
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.9),
            self._long_vote(SignalSource.PROMETHEUS, 0.8),
        ]
        coord.aggregate(votes)
        assert coord.decision_count == 1

    def test_reset(self):
        coord = self._make_coord()
        coord._decision_count = 10
        coord._sentinel_veto = True
        coord.reset()
        assert coord.decision_count == 0
        assert not coord._sentinel_veto

    def test_source_performance_analytics(self):
        coord = self._make_coord(
            min_consensus_score=0.1, min_agreement_ratio=0.3, min_confidence=0.1
        )
        votes = [
            self._long_vote(SignalSource.HYDRA, 0.9),
            self._long_vote(SignalSource.PROMETHEUS, 0.8),
        ]
        coord.aggregate(votes)
        perf = coord.get_source_performance()
        assert "HYDRA" in perf
        assert perf["HYDRA"]["total"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Reasoner
# ═══════════════════════════════════════════════════════════════════════════

class TestAresReasoner:
    def test_mock_resolution(self):
        r = AresReasoner(ReasonerConfig(provider=LLMProvider.MOCK))
        from aphelion.ares.coordinator import StrategyVote, SignalSource
        votes = [
            StrategyVote(source=SignalSource.HYDRA, direction=1, confidence=0.8),
            StrategyVote(source=SignalSource.PROMETHEUS, direction=-1, confidence=0.6),
        ]
        result = r.resolve_conflict(votes, "TRENDING")
        assert result is not None
        assert result.direction in (-1, 0, 1)
        assert 0 <= result.confidence <= 1

    def test_mock_thesis_generation(self):
        r = AresReasoner(ReasonerConfig(provider=LLMProvider.MOCK))
        thesis = r.generate_thesis(1, 0.75, {"atr": 5.0, "rsi_14": 65}, "TRENDING")
        assert isinstance(thesis, str)
        assert len(thesis) > 0
        assert "LONG" in thesis

    def test_rule_based_fallback(self):
        r = AresReasoner(ReasonerConfig(provider=LLMProvider.MOCK))
        from aphelion.ares.coordinator import StrategyVote, SignalSource
        votes = [
            StrategyVote(source=SignalSource.HYDRA, direction=1, confidence=0.9),
            StrategyVote(source=SignalSource.PROMETHEUS, direction=1, confidence=0.8),
        ]
        result = r._rule_based_fallback(votes)
        assert result.direction == 1
        assert "Long" in result.reasoning

    def test_rule_based_no_consensus(self):
        r = AresReasoner()
        from aphelion.ares.coordinator import StrategyVote, SignalSource
        votes = [
            StrategyVote(source=SignalSource.HYDRA, direction=1, confidence=0.5),
            StrategyVote(source=SignalSource.PROMETHEUS, direction=-1, confidence=0.5),
        ]
        result = r._rule_based_fallback(votes)
        assert result.direction == 0

    def test_parse_json_response(self):
        r = AresReasoner()
        content = '{"direction": -1, "confidence": 0.72, "reasoning": "Bearish trend"}'
        result = r._parse_llm_response(content, LLMProvider.OPENAI, tokens=100)
        assert result.direction == -1
        assert result.confidence == pytest.approx(0.72)
        assert "Bearish" in result.reasoning

    def test_parse_text_fallback(self):
        r = AresReasoner()
        content = "I recommend going LONG on gold due to strong support."
        result = r._parse_llm_response(content, LLMProvider.OPENAI)
        assert result.direction == 1

    def test_stats_tracking(self):
        r = AresReasoner(ReasonerConfig(provider=LLMProvider.MOCK))
        from aphelion.ares.coordinator import StrategyVote, SignalSource
        votes = [
            StrategyVote(source=SignalSource.HYDRA, direction=1, confidence=0.8),
        ]
        r.resolve_conflict(votes)
        assert r.call_count == 1
        assert r.total_tokens > 0

    def test_empty_votes_fallback(self):
        r = AresReasoner()
        result = r._rule_based_fallback([])
        assert result.direction == 0


# ═══════════════════════════════════════════════════════════════════════════
# Decision Journal
# ═══════════════════════════════════════════════════════════════════════════

class TestDecisionJournal:
    def test_record_entry(self):
        j = DecisionJournal()
        entry = j.record(
            direction=1, consensus_score=0.7, confidence=0.8,
            agreement_ratio=0.9, reasoning="Bull run",
        )
        assert entry.entry_id == "ARES-000001"
        assert j.entry_count == 1

    def test_record_outcome(self):
        j = DecisionJournal()
        entry = j.record(1, 0.7, 0.8, 0.9, "Bull")
        j.record_outcome(entry.entry_id, pnl=150.0, direction_correct=True)
        assert entry.outcome_pnl == 150.0
        assert entry.outcome_direction_correct is True

    def test_accuracy_tracking(self):
        j = DecisionJournal()
        e1 = j.record(1, 0.7, 0.8, 0.9, "Bull")
        e2 = j.record(-1, 0.6, 0.7, 0.8, "Bear")
        e3 = j.record(1, 0.5, 0.6, 0.7, "Bull2")
        j.record_outcome(e1.entry_id, 100, True)
        j.record_outcome(e2.entry_id, -50, False)
        j.record_outcome(e3.entry_id, 80, True)
        assert j.accuracy() == pytest.approx(2 / 3)

    def test_accuracy_no_outcomes(self):
        j = DecisionJournal()
        j.record(1, 0.7, 0.8, 0.9, "Bull")
        assert j.accuracy() == 0.0

    def test_persistence_jsonl(self, tmp_path):
        path = tmp_path / "test_journal.jsonl"
        j = DecisionJournal(path)
        j.record(1, 0.7, 0.8, 0.9, "Bull")
        j.record(-1, 0.5, 0.6, 0.7, "Bear")

        # Verify file written
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        d = json.loads(lines[0])
        assert d["direction"] == 1

    def test_load_from_file(self, tmp_path):
        path = tmp_path / "test_journal.jsonl"
        j1 = DecisionJournal(path)
        j1.record(1, 0.7, 0.8, 0.9, "Bull")
        j1.record(-1, 0.5, 0.6, 0.7, "Bear")

        j2 = DecisionJournal()
        count = j2.load(path)
        assert count == 2
        assert j2.entry_count == 2

    def test_recent_entries(self):
        j = DecisionJournal()
        for i in range(10):
            j.record(1, 0.5 + i * 0.01, 0.6, 0.7, f"Entry {i}")
        recent = j.recent(3)
        assert len(recent) == 3
        assert "Entry 9" in recent[-1].reasoning

    def test_average_confidence(self):
        j = DecisionJournal()
        j.record(1, 0.7, 0.80, 0.9, "Bull")
        j.record(-1, 0.5, 0.60, 0.7, "Bear")
        assert j.average_confidence() == pytest.approx(0.70)

    def test_clear(self):
        j = DecisionJournal()
        j.record(1, 0.7, 0.8, 0.9, "Bull")
        j.clear()
        assert j.entry_count == 0

    def test_entry_serialisation(self):
        entry = JournalEntry(
            entry_id="TEST-001",
            timestamp=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
            direction=1, consensus_score=0.7,
            confidence=0.8, agreement_ratio=0.9,
            reasoning="Test thesis",
        )
        d = entry.to_dict()
        e2 = JournalEntry.from_dict(d)
        assert e2.entry_id == "TEST-001"
        assert e2.direction == 1
        assert e2.confidence == pytest.approx(0.8)
