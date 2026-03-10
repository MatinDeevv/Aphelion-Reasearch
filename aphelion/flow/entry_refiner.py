"""
OMEGA — Entry Refiner
Phase 17 — Engineering Spec v3.0

Refines entry timing for OMEGA swing trades using pullback analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EntrySetup:
    """Refined entry parameters."""
    valid: bool
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reason: str = ""


class EntryRefiner:
    """
    Refines OMEGA entries by waiting for optimal pullback into trend.
    Uses ATR-based stop/target placement.
    """

    def __init__(
        self,
        pullback_threshold: float = 0.003,
        atr_sl_mult: float = 3.0,
        atr_tp_mult: float = 5.0,
        min_rr: float = 2.0,
    ):
        self._pullback = pullback_threshold
        self._sl_mult = atr_sl_mult
        self._tp_mult = atr_tp_mult
        self._min_rr = min_rr

    def evaluate_long(
        self, current_price: float, ema_fast: float, atr: float
    ) -> EntrySetup:
        """Evaluate long entry on pullback to EMA."""
        pullback = (ema_fast - current_price) / ema_fast if ema_fast > 0 else 0

        if pullback < self._pullback:
            return EntrySetup(False, 0, 0, 0, 0, "Pullback insufficient")

        sl = current_price - atr * self._sl_mult
        tp = current_price + atr * self._tp_mult
        risk = current_price - sl
        rr = (tp - current_price) / risk if risk > 0 else 0

        if rr < self._min_rr:
            return EntrySetup(False, current_price, sl, tp, rr, "RR too low")

        return EntrySetup(True, current_price, sl, tp, rr, "LONG_PULLBACK_ENTRY")

    def evaluate_short(
        self, current_price: float, ema_fast: float, atr: float
    ) -> EntrySetup:
        """Evaluate short entry on retracement above EMA."""
        pullback = (current_price - ema_fast) / ema_fast if ema_fast > 0 else 0

        if pullback < self._pullback:
            return EntrySetup(False, 0, 0, 0, 0, "Pullback insufficient")

        sl = current_price + atr * self._sl_mult
        tp = current_price - atr * self._tp_mult
        risk = sl - current_price
        rr = (current_price - tp) / risk if risk > 0 else 0

        if rr < self._min_rr:
            return EntrySetup(False, current_price, sl, tp, rr, "RR too low")

        return EntrySetup(True, current_price, sl, tp, rr, "SHORT_PULLBACK_ENTRY")
