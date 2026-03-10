"""
OMEGA — Exit Manager
Phase 17 — Engineering Spec v3.0

Manages exits for OMEGA swing trades: trailing stops, break-even moves,
and time-based exits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    """Exit decision for an open position."""
    should_exit: bool
    reason: str = ""
    new_stop: Optional[float] = None


class ExitManager:
    """
    Manages OMEGA position exits.

    Strategies:
    1. ATR trailing stop
    2. Break-even protection (move SL to entry + spread after 1 ATR profit)
    3. Time-based exit (max hold time for swing trades)
    4. Trend reversal exit
    """

    def __init__(
        self,
        trail_atr_mult: float = 2.5,
        breakeven_trigger_atr: float = 1.0,
        max_hold_bars: int = 48,  # 48 H1 bars = 2 days
    ):
        self._trail_mult = trail_atr_mult
        self._be_trigger = breakeven_trigger_atr
        self._max_hold = max_hold_bars

    def evaluate(
        self,
        direction: int,
        entry_price: float,
        current_price: float,
        current_stop: float,
        atr: float,
        bars_held: int,
        trend_still_valid: bool = True,
    ) -> ExitDecision:
        """Evaluate whether to modify or exit a position."""

        # 1. Trend reversal
        if not trend_still_valid:
            return ExitDecision(True, "TREND_REVERSED")

        # 2. Time-based exit
        if bars_held >= self._max_hold:
            return ExitDecision(True, f"MAX_HOLD ({self._max_hold} bars)")

        # 3. Calculate new trailing stop
        if direction == 1:  # LONG
            new_trail = current_price - atr * self._trail_mult
            # Only move stop up
            new_stop = max(current_stop, new_trail)

            # Break-even protection
            profit_atr = (current_price - entry_price) / atr if atr > 0 else 0
            if profit_atr >= self._be_trigger:
                be_stop = entry_price + atr * 0.1  # Entry + tiny buffer
                new_stop = max(new_stop, be_stop)

            if new_stop != current_stop:
                return ExitDecision(False, "TRAIL_UPDATED", new_stop)

        elif direction == -1:  # SHORT
            new_trail = current_price + atr * self._trail_mult
            new_stop = min(current_stop, new_trail)

            profit_atr = (entry_price - current_price) / atr if atr > 0 else 0
            if profit_atr >= self._be_trigger:
                be_stop = entry_price - atr * 0.1
                new_stop = min(new_stop, be_stop)

            if new_stop != current_stop:
                return ExitDecision(False, "TRAIL_UPDATED", new_stop)

        return ExitDecision(False, "HOLD")
