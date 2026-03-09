"""
APHELION ARES — LLM Brain & Strategy Coordinator (Phase 10)

ARES is the Council-tier intelligence module that:
  1. Aggregates signals from HYDRA, PROMETHEUS, and Money Maker strategies
  2. Applies LLM-powered reasoning to resolve conflicting signals
  3. Makes final trade/no-trade decisions with natural language explanations
  4. Coordinates strategy activation based on market regime
  5. Maintains a decision journal for audit and learning

Architecture:
  coordinator — Signal aggregation and conflict resolution
  reasoner    — LLM reasoning engine (pluggable: local/API)
  journal     — Decision audit log with explanations
  prompts/    — System and decision prompts
"""

from aphelion.ares.coordinator import (
    AresConfig,
    SignalSource,
    AggregatedSignal,
    StrategyVote,
    AresCoordinator,
)
from aphelion.ares.reasoner import (
    ReasonerConfig,
    AresReasoner,
    ReasoningResult,
)
from aphelion.ares.journal import (
    JournalEntry,
    DecisionJournal,
)

__all__ = [
    "AresConfig", "SignalSource", "AggregatedSignal", "StrategyVote",
    "AresCoordinator",
    "ReasonerConfig", "AresReasoner", "ReasoningResult",
    "JournalEntry", "DecisionJournal",
]
