"""Trade proposal validation against immutable SENTINEL rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from aphelion.core.clock import MarketClock
from aphelion.core.config import SENTINEL, SYMBOL
from aphelion.risk.sentinel.core import SentinelCore


@dataclass
class TradeProposal:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    size_pct: float  # Proposed fraction of account
    proposed_by: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationResult:
    approved: bool
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adjusted_size_pct: float = 0.0


class TradeValidator:
    """Validates all trade proposals against hardcoded SENTINEL constraints."""

    def __init__(self, sentinel_core: SentinelCore, clock: MarketClock):
        self._sentinel = sentinel_core
        self._clock = clock

    def validate(self, proposal: TradeProposal) -> ValidationResult:
        rejections: list[str] = []
        warnings: list[str] = []

        # Check 1: Trading allowed
        if not self._sentinel.is_trading_allowed():
            reason = ""
            if self._sentinel.l3_triggered:
                reason = "L3_DISCONNECT"
            elif self._clock.is_news_lockout():
                reason = "NEWS_LOCKOUT"
            elif self._clock.is_friday_lockout():
                reason = "FRIDAY_LOCKOUT"
            elif not self._clock.is_market_open():
                reason = "MARKET_CLOSED"
            rejections.append(f"TRADING_HALTED: {reason}")

        # Check 2: Stop loss present and valid direction placement
        if proposal.stop_loss <= 0:
            rejections.append("NO_STOP_LOSS: stop_loss must be > 0")
        if proposal.direction == "LONG" and proposal.stop_loss >= proposal.entry_price:
            rejections.append("INVALID_STOP_LOSS: LONG stop_loss must be below entry")
        if proposal.direction == "SHORT" and proposal.stop_loss <= proposal.entry_price:
            rejections.append("INVALID_STOP_LOSS: SHORT stop_loss must be above entry")

        # Check 3: Risk:Reward ratio
        risk = abs(proposal.entry_price - proposal.stop_loss)
        reward = abs(proposal.take_profit - proposal.entry_price)
        rr = reward / risk if risk > 0 else 0.0
        if rr < SENTINEL.min_risk_reward:
            rejections.append(f"INSUFFICIENT_RR: {rr:.2f} < {SENTINEL.min_risk_reward}")

        # Check 4: Position count
        if self._sentinel.get_open_position_count() >= SENTINEL.max_simultaneous_positions:
            rejections.append(
                f"MAX_POSITIONS: {SENTINEL.max_simultaneous_positions} already open"
            )

        # Check 5: Position size
        if proposal.size_pct > SENTINEL.max_position_pct:
            rejections.append(
                f"SIZE_EXCEEDED: {proposal.size_pct:.3f} > {SENTINEL.max_position_pct}"
            )

        # Check 6: Total exposure
        total = self._sentinel.get_total_exposure_pct() + proposal.size_pct
        max_total = SENTINEL.max_position_pct * SENTINEL.max_simultaneous_positions
        if total > max_total:
            rejections.append(f"EXPOSURE_EXCEEDED: {total:.3f} > {max_total:.3f}")

        # Check 7: Symbol
        if proposal.symbol != SYMBOL:
            rejections.append(f"INVALID_SYMBOL: {proposal.symbol}  only {SYMBOL} supported")

        approved = len(rejections) == 0
        adjusted_size_pct = proposal.size_pct if approved else 0.0
        return ValidationResult(
            approved=approved,
            rejections=rejections,
            warnings=warnings,
            adjusted_size_pct=adjusted_size_pct,
        )

    def bulk_validate(self, proposals: list[TradeProposal]) -> list[ValidationResult]:
        return [self.validate(proposal) for proposal in proposals]
