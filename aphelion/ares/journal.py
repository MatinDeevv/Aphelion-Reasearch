"""
APHELION ARES — Decision Journal

Persistent audit log for every ARES decision. Records:
  - All strategy votes and their weights
  - The aggregated signal and final direction
  - LLM reasoning (if used)
  - Market context at decision time
  - Outcome tracking (linked to trade results)

Used for:
  - Regulatory audit compliance
  - Strategy refinement and learning
  - Post-session review
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class JournalEntry:
    """A single decision record in the ARES journal."""
    entry_id: str
    timestamp: datetime
    # Decision
    direction: int                           # -1, 0, 1
    consensus_score: float
    confidence: float
    agreement_ratio: float
    reasoning: str
    # Context
    regime: str = "UNKNOWN"
    price: float = 0.0
    atr: float = 0.0
    session: str = ""
    # Votes
    votes: list[dict] = field(default_factory=list)
    # Outcome (filled after trade closes)
    outcome_pnl: Optional[float] = None
    outcome_direction_correct: Optional[bool] = None
    outcome_filled: bool = False

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "consensus_score": self.consensus_score,
            "confidence": self.confidence,
            "agreement_ratio": self.agreement_ratio,
            "reasoning": self.reasoning,
            "regime": self.regime,
            "price": self.price,
            "atr": self.atr,
            "session": self.session,
            "votes": self.votes,
            "outcome_pnl": self.outcome_pnl,
            "outcome_direction_correct": self.outcome_direction_correct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JournalEntry:
        return cls(
            entry_id=d["entry_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            direction=d["direction"],
            consensus_score=d["consensus_score"],
            confidence=d["confidence"],
            agreement_ratio=d["agreement_ratio"],
            reasoning=d["reasoning"],
            regime=d.get("regime", ""),
            price=d.get("price", 0.0),
            atr=d.get("atr", 0.0),
            session=d.get("session", ""),
            votes=d.get("votes", []),
            outcome_pnl=d.get("outcome_pnl"),
            outcome_direction_correct=d.get("outcome_direction_correct"),
        )


class DecisionJournal:
    """
    Persistent decision journal for ARES.
    Stores decision records as newline-delimited JSON (JSONL format)
    for fast append and streaming reads.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self._path = Path(path) if path else None
        self._entries: list[JournalEntry] = []
        self._counter: int = 0

    # ── Record ───────────────────────────────────────────────────────────────

    def record(
        self,
        direction: int,
        consensus_score: float,
        confidence: float,
        agreement_ratio: float,
        reasoning: str,
        votes: Optional[list[dict]] = None,
        regime: str = "",
        price: float = 0.0,
        atr: float = 0.0,
        session: str = "",
    ) -> JournalEntry:
        """Record a new decision."""
        self._counter += 1
        entry = JournalEntry(
            entry_id=f"ARES-{self._counter:06d}",
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            consensus_score=consensus_score,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            reasoning=reasoning,
            votes=votes or [],
            regime=regime,
            price=price,
            atr=atr,
            session=session,
        )
        self._entries.append(entry)

        # Append to file
        if self._path:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception:
                logger.warning("Failed to write journal entry", exc_info=True)

        return entry

    # ── Outcome Tracking ─────────────────────────────────────────────────────

    def record_outcome(
        self, entry_id: str, pnl: float, direction_correct: bool,
    ) -> None:
        """Link a trade outcome back to its decision."""
        for entry in reversed(self._entries):
            if entry.entry_id == entry_id:
                entry.outcome_pnl = pnl
                entry.outcome_direction_correct = direction_correct
                entry.outcome_filled = True
                break

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def entries(self) -> list[JournalEntry]:
        return list(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def recent(self, n: int = 20) -> list[JournalEntry]:
        return self._entries[-n:]

    def accuracy(self) -> float:
        """Compute directional accuracy of all outcomes-tracked decisions."""
        tracked = [e for e in self._entries if e.outcome_filled]
        if not tracked:
            return 0.0
        correct = sum(1 for e in tracked if e.outcome_direction_correct)
        return correct / len(tracked)

    def average_confidence(self) -> float:
        if not self._entries:
            return 0.0
        return sum(e.confidence for e in self._entries) / len(self._entries)

    # ── Load from File ───────────────────────────────────────────────────────

    def load(self, path: Optional[str | Path] = None) -> int:
        """Load entries from a JSONL file. Returns count loaded."""
        p = Path(path) if path else self._path
        if p is None or not p.exists():
            return 0

        count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = JournalEntry.from_dict(json.loads(line))
                    self._entries.append(entry)
                    count += 1
                except Exception:
                    logger.warning("Failed to parse journal line", exc_info=True)

        self._counter = count
        return count

    def clear(self) -> None:
        """Clear in-memory entries."""
        self._entries.clear()
        self._counter = 0
