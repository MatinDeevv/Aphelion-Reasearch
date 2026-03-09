"""
APHELION ARES — Strategy Coordinator

Aggregates signals from multiple intelligence sources (HYDRA, PROMETHEUS,
Money Makers) and resolves conflicts using weighted voting, regime awareness,
and optional LLM reasoning.

The coordinator implements the spec's tiered governance voting model:
  - Council-tier modules (SENTINEL, ARES) get 100 votes
  - Minister-tier (HYDRA, PROMETHEUS) get 40 votes
  - Commander-tier (strategies) get 10 votes

Final decision respects SENTINEL veto power.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

from aphelion.core.config import SENTINEL, Tier, TIER_VOTE_WEIGHTS

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Source of a trading signal."""
    HYDRA = "HYDRA"
    PROMETHEUS = "PROMETHEUS"
    VENOM = "VENOM"
    REAPER = "REAPER"
    APEX = "APEX"
    WRAITH = "WRAITH"
    MANUAL = "MANUAL"


@dataclass
class StrategyVote:
    """A single strategy's vote on trade direction."""
    source: SignalSource
    direction: int               # -1=SHORT, 0=FLAT, 1=LONG
    confidence: float            # [0, 1]
    tier: Tier = Tier.COMMANDER
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Direction * confidence * tier vote weight."""
        weight = TIER_VOTE_WEIGHTS.get(self.tier, 1)
        if weight == float('inf'):
            weight = 1000  # Cap for arithmetic
        return self.direction * self.confidence * weight


@dataclass
class AggregatedSignal:
    """Result of aggregating multiple strategy votes."""
    direction: int                           # -1, 0, 1
    consensus_score: float                   # Normalised weighted score [-1, 1]
    confidence: float                        # Overall confidence [0, 1]
    agreement_ratio: float                   # Fraction of votes agreeing with direction
    votes: list[StrategyVote] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vetoed: bool = False
    veto_reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and not self.vetoed and self.confidence > 0


@dataclass
class AresConfig:
    """ARES coordinator configuration."""
    # Consensus
    min_consensus_score: float = 0.20        # Min weighted score to trade
    min_agreement_ratio: float = 0.50        # Min fraction agreeing
    min_confidence: float = 0.55             # Min overall confidence
    # Regime
    use_regime_filter: bool = True
    # LLM
    use_llm_reasoning: bool = False          # Enable LLM for conflict resolution
    llm_conflict_threshold: float = 0.30     # Trigger LLM when agreement < this
    # Strategy activation
    max_active_strategies: int = 4
    auto_deactivate_losing: bool = True
    deactivation_sharpe_threshold: float = -0.5
    # Cooldown
    decision_cooldown_bars: int = 3


class AresCoordinator:
    """
    ARES strategy coordination engine.

    Collects votes from all active strategies, computes weighted consensus,
    optionally invokes LLM reasoning for conflicts, and produces a final
    aggregated signal for execution.
    """

    def __init__(
        self,
        config: Optional[AresConfig] = None,
        reasoner: Optional[object] = None,   # AresReasoner (optional)
    ):
        self._config = config or AresConfig()
        self._reasoner = reasoner
        self._active_sources: set[SignalSource] = {
            SignalSource.HYDRA,
            SignalSource.PROMETHEUS,
        }
        self._bars_since_decision: int = 999
        self._decision_count: int = 0
        self._decision_history: list[AggregatedSignal] = []
        self._regime: str = "UNKNOWN"
        self._sentinel_veto: bool = False

    # ── Signal Collection ────────────────────────────────────────────────────

    def aggregate(self, votes: list[StrategyVote]) -> AggregatedSignal:
        """
        Aggregate strategy votes into a single trading decision.

        Process:
          1. Filter to active sources
          2. Compute weighted consensus
          3. Check conflict → optionally invoke LLM
          4. Apply regime filter
          5. Check SENTINEL veto
          6. Return final decision
        """
        self._bars_since_decision += 1

        # Cooldown
        if self._bars_since_decision < self._config.decision_cooldown_bars:
            return AggregatedSignal(direction=0, consensus_score=0.0,
                                   confidence=0.0, agreement_ratio=0.0,
                                   reasoning="Cooldown active")

        # Filter to active sources
        active_votes = [v for v in votes if v.source in self._active_sources]

        if not active_votes:
            return AggregatedSignal(direction=0, consensus_score=0.0,
                                   confidence=0.0, agreement_ratio=0.0,
                                   reasoning="No active votes")

        # 1. Weighted consensus
        total_weight = sum(abs(v.weighted_score) for v in active_votes)
        if total_weight < 1e-10:
            return AggregatedSignal(direction=0, consensus_score=0.0,
                                   confidence=0.0, agreement_ratio=0.0,
                                   votes=active_votes, reasoning="Zero weight")

        weighted_sum = sum(v.weighted_score for v in active_votes)
        consensus = weighted_sum / total_weight  # [-1, 1]

        # 2. Direction
        if consensus > self._config.min_consensus_score:
            direction = 1
        elif consensus < -self._config.min_consensus_score:
            direction = -1
        else:
            direction = 0

        # 3. Agreement
        if direction != 0:
            agreeing = [v for v in active_votes if v.direction == direction]
        else:
            agreeing = [v for v in active_votes if v.direction == 0]
        agreement_ratio = len(agreeing) / len(active_votes) if active_votes else 0.0

        # 4. Overall confidence
        confidence_scores = [v.confidence for v in active_votes]
        confidence = float(np.mean(confidence_scores)) * abs(consensus)

        reasoning_parts = []

        # 5. Conflict → LLM reasoning (if enabled)
        if (self._config.use_llm_reasoning
                and self._reasoner is not None
                and agreement_ratio < self._config.llm_conflict_threshold):
            try:
                llm_result = self._reasoner.resolve_conflict(active_votes, self._regime)
                if llm_result is not None:
                    direction = llm_result.direction
                    confidence = llm_result.confidence
                    reasoning_parts.append(f"LLM override: {llm_result.reasoning}")
            except Exception:
                logger.warning("LLM reasoning failed", exc_info=True)
                reasoning_parts.append("LLM reasoning failed — using vote consensus")

        # 6. Regime filter
        if self._config.use_regime_filter and self._regime == "VOL_EXPANSION":
            confidence *= 0.7  # Reduce confidence in volatile regimes
            reasoning_parts.append("Vol-expansion regime penalty applied")

        # 7. SENTINEL veto check
        vetoed = False
        veto_reason = ""
        if self._sentinel_veto:
            vetoed = True
            veto_reason = "SENTINEL veto active"
            reasoning_parts.append(veto_reason)

        # 8. Minimum thresholds
        if agreement_ratio < self._config.min_agreement_ratio:
            direction = 0
            reasoning_parts.append(f"Agreement {agreement_ratio:.0%} below threshold")

        if confidence < self._config.min_confidence and direction != 0:
            direction = 0
            reasoning_parts.append(f"Confidence {confidence:.2f} below threshold")

        signal = AggregatedSignal(
            direction=direction,
            consensus_score=float(consensus),
            confidence=float(confidence),
            agreement_ratio=float(agreement_ratio),
            votes=active_votes,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else "Consensus trade",
            vetoed=vetoed,
            veto_reason=veto_reason,
        )

        if direction != 0:
            self._bars_since_decision = 0
            self._decision_count += 1

        self._decision_history.append(signal)
        # Keep bounded
        if len(self._decision_history) > 1000:
            self._decision_history = self._decision_history[-500:]

        return signal

    # ── Regime ───────────────────────────────────────────────────────────────

    def set_regime(self, regime: str) -> None:
        """Update the current market regime."""
        self._regime = regime

    # ── Source Management ────────────────────────────────────────────────────

    def activate_source(self, source: SignalSource) -> None:
        self._active_sources.add(source)

    def deactivate_source(self, source: SignalSource) -> None:
        self._active_sources.discard(source)

    @property
    def active_sources(self) -> set[SignalSource]:
        return set(self._active_sources)

    # ── SENTINEL Interface ───────────────────────────────────────────────────

    def set_sentinel_veto(self, veto: bool) -> None:
        """SENTINEL can veto all trading via this flag."""
        self._sentinel_veto = veto

    # ── Analytics ────────────────────────────────────────────────────────────

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def recent_decisions(self) -> list[AggregatedSignal]:
        return self._decision_history[-50:]

    def get_source_performance(self) -> dict[str, dict]:
        """Analyse vote accuracy for each source."""
        source_stats: dict[str, dict] = {}
        for signal in self._decision_history:
            for vote in signal.votes:
                src = vote.source.value
                if src not in source_stats:
                    source_stats[src] = {"total": 0, "correct": 0, "confidence_sum": 0.0}
                source_stats[src]["total"] += 1
                source_stats[src]["confidence_sum"] += vote.confidence
                if vote.direction == signal.direction:
                    source_stats[src]["correct"] += 1

        for src, stats in source_stats.items():
            total = stats["total"]
            stats["accuracy"] = stats["correct"] / total if total > 0 else 0.0
            stats["avg_confidence"] = stats["confidence_sum"] / total if total > 0 else 0.0

        return source_stats

    def reset(self) -> None:
        """Reset coordinator state."""
        self._bars_since_decision = 999
        self._decision_count = 0
        self._decision_history.clear()
        self._sentinel_veto = False
