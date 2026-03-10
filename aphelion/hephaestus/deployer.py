"""
HEPHAESTUS — Deployer

Wires validated strategies into the live ARES council and manages
shadow-mode tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from aphelion.hephaestus.models import (
    ForgedStrategy,
    ShadowEvaluation,
    StrategySpec,
    ValidationReport,
    Vote,
)

logger = logging.getLogger(__name__)


# ─── Shadow-mode tracker ────────────────────────────────────────────────────


@dataclass
class ShadowRecord:
    """A single shadow-mode vote and its actual outcome."""
    voter_id: str
    direction: int
    confidence: float
    outcome: int  # 1=win, -1=loss, 0=scratch
    r_multiple: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ShadowModeTracker:
    """Tracks a strategy in shadow mode — votes without power.

    Automatically evaluates for promotion or rejection after
    ``SHADOW_TRADE_THRESHOLD`` trades.
    """

    SHADOW_TRADE_THRESHOLD: int = 500
    PERFORMANCE_REGRESSION_LIMIT: float = 0.20

    def __init__(self) -> None:
        self._records: dict[str, list[ShadowRecord]] = {}
        self._validated_sharpes: dict[str, float] = {}

    def register(self, voter_id: str, validated_sharpe: float) -> None:
        """Start tracking a voter in shadow mode."""
        self._records[voter_id] = []
        self._validated_sharpes[voter_id] = validated_sharpe

    def record_shadow_vote(
        self,
        voter_id: str,
        vote: Vote,
        outcome: int,
        r_multiple: float = 0.0,
    ) -> None:
        """Record a shadow vote and its actual outcome."""
        if voter_id not in self._records:
            return
        self._records[voter_id].append(
            ShadowRecord(
                voter_id=voter_id,
                direction=vote.direction,
                confidence=vote.confidence,
                outcome=outcome,
                r_multiple=r_multiple,
            )
        )

    def trade_count(self, voter_id: str) -> int:
        """Number of shadow trades recorded."""
        return len(self._records.get(voter_id, []))

    def is_ready_for_evaluation(self, voter_id: str) -> bool:
        """Whether enough shadow trades have been recorded."""
        return self.trade_count(voter_id) >= self.SHADOW_TRADE_THRESHOLD

    def evaluate_for_promotion(self, voter_id: str) -> ShadowEvaluation:
        """Evaluate a shadow voter for promotion.

        Returns PROMOTE, CONTINUE_SHADOW, or REJECT.
        """
        records = self._records.get(voter_id, [])
        if len(records) < self.SHADOW_TRADE_THRESHOLD:
            return ShadowEvaluation.CONTINUE_SHADOW

        shadow_sharpe = self._compute_shadow_sharpe(records)
        validated_sharpe = self._validated_sharpes.get(voter_id, 0.0)

        if validated_sharpe <= 0:
            # No backtest baseline — evaluate on absolute merit
            if shadow_sharpe >= 1.0:
                return ShadowEvaluation.PROMOTE
            return ShadowEvaluation.REJECT

        regression = (validated_sharpe - shadow_sharpe) / validated_sharpe

        if regression > self.PERFORMANCE_REGRESSION_LIMIT:
            return ShadowEvaluation.REJECT
        elif shadow_sharpe >= validated_sharpe * 0.8:
            return ShadowEvaluation.PROMOTE
        else:
            return ShadowEvaluation.CONTINUE_SHADOW

    @staticmethod
    def _compute_shadow_sharpe(records: list[ShadowRecord]) -> float:
        """Simplified Sharpe from R-multiples."""
        if not records:
            return 0.0
        r_values = np.array([r.r_multiple for r in records], dtype=np.float64)
        if len(r_values) < 2:
            return 0.0
        mean = float(np.mean(r_values))
        std = float(np.std(r_values, ddof=1))
        if std < 1e-10:
            # Near-zero variance: sign of mean determines result
            return 100.0 if mean > 0 else (-100.0 if mean < 0 else 0.0)
        return mean / std * np.sqrt(252)


# ─── Deployer ────────────────────────────────────────────────────────────────


class HephaestusDeployer:
    """Wires a validated strategy into the live ARES council.

    Supports shadow-mode deployment, promotion, and revocation.
    """

    def __init__(
        self,
        ares_council: Optional[object] = None,
        shadow_tracker: Optional[ShadowModeTracker] = None,
    ) -> None:
        self._ares = ares_council
        self._shadow = shadow_tracker or ShadowModeTracker()
        self._deployed: dict[str, dict] = {}  # voter_id → metadata

    @property
    def shadow_tracker(self) -> ShadowModeTracker:
        return self._shadow

    # ── Public API ───────────────────────────────────────────────────────

    def deploy(
        self,
        forged: ForgedStrategy,
        spec: StrategySpec,
        validation: ValidationReport,
        mode: str = "SHADOW",
    ) -> str:
        """Deploy a validated strategy.

        Returns a unique voter ID string.
        """
        voter_id = f"heph_{spec.name.lower().replace(' ', '_')[:20]}_{id(forged) % 0xFFFF:04x}"

        metadata = {
            "source": "HEPHAESTUS",
            "class_name": forged.class_name,
            "validated_sharpe": validation.sharpe_ratio,
            "validation_date": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "spec_name": spec.name,
            "python_code": forged.python_code,
        }
        self._deployed[voter_id] = metadata

        # Register shadow tracking
        if mode == "SHADOW":
            self._shadow.register(voter_id, validation.sharpe_ratio)

        logger.info(
            "HEPHAESTUS deployed %s (%s) in %s mode",
            voter_id,
            spec.name,
            mode,
        )
        return voter_id

    def promote(self, voter_id: str) -> bool:
        """Promote a shadow voter to full deployment."""
        if voter_id not in self._deployed:
            return False
        self._deployed[voter_id]["mode"] = "FULL"
        logger.info("HEPHAESTUS promoted %s to FULL", voter_id)
        return True

    def revoke(self, voter_id: str, reason: str = "") -> bool:
        """Revoke and remove a deployed voter."""
        if voter_id not in self._deployed:
            return False
        self._deployed.pop(voter_id)
        logger.info("HEPHAESTUS revoked %s: %s", voter_id, reason)
        return True

    def list_deployed(self) -> list[dict]:
        """Return metadata for all deployed voters."""
        return [
            {"voter_id": vid, **meta}
            for vid, meta in self._deployed.items()
        ]

    def get_status(self, voter_id: str) -> Optional[dict]:
        """Get deployment metadata for a voter."""
        meta = self._deployed.get(voter_id)
        if meta is None:
            return None
        return {"voter_id": voter_id, **meta}
