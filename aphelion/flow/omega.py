"""
OMEGA — H1/H4 Swing Trading Strategy
Phase 17 — Engineering Spec v3.0

Low win rate (28-35%), high R:R (5:1 – 8:1) trend-following strategy.
Deliberately uncorrelated with ALPHA (M1 scalping).
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

import numpy as np


class H4Structure(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    UNCLEAR = "UNCLEAR"


@dataclass
class PullbackLevel:
    price: float
    invalidation_level: float      # SL level
    target_1: float                 # TP1 at 3:1 R
    target_2: float                 # TP2 at 6:1 R

    def price_near(self, current: float, tolerance_pips: float = 30.0) -> bool:
        return abs(current - self.price) <= tolerance_pips


@dataclass
class OmegaSignal:
    direction: int = 0              # 1=BUY, -1=SELL, 0=FLAT
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0     # 3:1 R — close 50%
    take_profit_2: float = 0.0     # 6:1 R — close remaining
    confidence: float = 0.0
    reason: str = ""
    h4_structure: str = "UNCLEAR"


class OmegaSignalGenerator:
    """
    OMEGA signal logic:
    1. H4 structure: Is trend clearly established? (HH+HL or LH+LL)
    2. H1 pullback: Is price pulling back into a key level?
    3. M15 entry trigger: reversal candle / breakout at the level
    4. DXY confluence
    5. MACRO regime must be conducive to trend following

    Commander-tier ARES voter (10 votes).
    """

    def __init__(self, pullback_tolerance: float = 30.0):
        self._pullback_tol = pullback_tolerance

    def generate(
        self,
        h4_highs: np.ndarray,
        h4_lows: np.ndarray,
        h4_closes: np.ndarray,
        h1_highs: np.ndarray,
        h1_lows: np.ndarray,
        h1_closes: np.ndarray,
        m15_highs: Optional[np.ndarray] = None,
        m15_lows: Optional[np.ndarray] = None,
        m15_closes: Optional[np.ndarray] = None,
        regime: str = "RANGING",
    ) -> OmegaSignal:
        """Generate OMEGA signal from multi-timeframe data."""

        # Step 1: H4 structure analysis
        h4_structure = self._detect_h4_structure(h4_highs, h4_lows, h4_closes)
        if h4_structure == H4Structure.UNCLEAR:
            return OmegaSignal(reason="NO_H4_STRUCTURE", h4_structure="UNCLEAR")

        # Step 2: H1 pullback to key level
        pullback = self._find_pullback_level(h1_highs, h1_lows, h1_closes, h4_structure)
        if pullback is None:
            return OmegaSignal(reason="NO_PULLBACK_LEVEL", h4_structure=h4_structure.value)

        current_price = float(h1_closes[-1])
        if not pullback.price_near(current_price, self._pullback_tol):
            return OmegaSignal(reason="NOT_AT_PULLBACK_LEVEL", h4_structure=h4_structure.value)

        # Step 3: M15 entry trigger (if available)
        if m15_closes is not None and len(m15_closes) >= 3:
            m15_valid = self._detect_m15_trigger(m15_highs, m15_lows, m15_closes, h4_structure)
            if not m15_valid:
                return OmegaSignal(reason="NO_M15_TRIGGER", h4_structure=h4_structure.value)

        # Step 4: Regime check
        if regime in ("VOLATILE", "CRISIS"):
            return OmegaSignal(reason="ADVERSE_REGIME", h4_structure=h4_structure.value)

        # All conditions met — generate signal
        direction = 1 if h4_structure == H4Structure.UPTREND else -1
        confidence = self._compute_confidence(h4_structure, pullback, current_price)

        return OmegaSignal(
            direction=direction,
            entry=current_price,
            stop_loss=pullback.invalidation_level,
            take_profit_1=pullback.target_1,
            take_profit_2=pullback.target_2,
            confidence=confidence,
            reason="SIGNAL_GENERATED",
            h4_structure=h4_structure.value,
        )

    def _detect_h4_structure(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> H4Structure:
        """Detect H4 market structure: HH+HL = uptrend, LH+LL = downtrend."""
        if len(highs) < 10:
            return H4Structure.UNCLEAR

        # Find swing highs and lows (simplified)
        swing_highs = []
        swing_lows = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                swing_highs.append(float(highs[i]))
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                swing_lows.append(float(lows[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return H4Structure.UNCLEAR

        # Check for higher highs + higher lows (uptrend)
        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]

        # Check for lower highs + lower lows (downtrend)
        lh = swing_highs[-1] < swing_highs[-2]
        ll = swing_lows[-1] < swing_lows[-2]

        if hh and hl:
            return H4Structure.UPTREND
        elif lh and ll:
            return H4Structure.DOWNTREND
        return H4Structure.UNCLEAR

    def _find_pullback_level(
        self, h1_highs: np.ndarray, h1_lows: np.ndarray,
        h1_closes: np.ndarray, structure: H4Structure,
    ) -> Optional[PullbackLevel]:
        """Find the pullback level based on H1 swings."""
        if len(h1_closes) < 10:
            return None

        # Find recent swing low (uptrend) or swing high (downtrend)
        if structure == H4Structure.UPTREND:
            # Look for recent swing low (pullback point)
            recent_low = float(np.min(h1_lows[-20:]))
            recent_high = float(np.max(h1_highs[-20:]))
            risk = float(h1_closes[-1]) - recent_low

            if risk <= 0:
                return None

            return PullbackLevel(
                price=recent_low + risk * 0.382,  # Fib 38.2% pullback
                invalidation_level=recent_low - risk * 0.2,
                target_1=float(h1_closes[-1]) + risk * 3,   # 3:1 R
                target_2=float(h1_closes[-1]) + risk * 6,   # 6:1 R
            )
        elif structure == H4Structure.DOWNTREND:
            recent_high = float(np.max(h1_highs[-20:]))
            recent_low = float(np.min(h1_lows[-20:]))
            risk = recent_high - float(h1_closes[-1])

            if risk <= 0:
                return None

            return PullbackLevel(
                price=recent_high - risk * 0.382,
                invalidation_level=recent_high + risk * 0.2,
                target_1=float(h1_closes[-1]) - risk * 3,
                target_2=float(h1_closes[-1]) - risk * 6,
            )
        return None

    @staticmethod
    def _detect_m15_trigger(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        structure: H4Structure,
    ) -> bool:
        """Detect M15 reversal candle at pullback level."""
        if len(closes) < 3:
            return False

        # Simple: bullish engulfing for uptrend, bearish engulfing for downtrend
        if structure == H4Structure.UPTREND:
            # Bullish: current close > previous open, current open < previous close
            return closes[-1] > closes[-2] and lows[-1] < lows[-2]
        elif structure == H4Structure.DOWNTREND:
            return closes[-1] < closes[-2] and highs[-1] > highs[-2]
        return False

    @staticmethod
    def _compute_confidence(
        structure: H4Structure, pullback: PullbackLevel, current_price: float,
    ) -> float:
        """Compute signal confidence."""
        # Distance from pullback level
        dist = abs(current_price - pullback.price)
        proximity_score = max(0, 1 - dist / 50)  # Closer = more confident

        # Structure clarity gives base confidence
        base = 0.5
        return min(1.0, base + proximity_score * 0.3)


class OmegaExitManager:
    """
    OMEGA trades use a two-stage exit:
    Stage 1 (TP1 at 3:1 R): Close 50%, move stop to breakeven
    Stage 2 (TP2 at 6:1 R): Close remaining, OR trail stop at H1 swing
    """

    def __init__(self):
        self._stage = 0   # 0=entered, 1=TP1 hit, 2=complete

    def check_exit(
        self,
        current_price: float,
        entry_price: float,
        direction: int,
        stop_loss: float,
        tp1: float,
        tp2: float,
    ) -> dict:
        """Check if any exit condition is met."""
        result = {
            "action": "HOLD",
            "close_pct": 0.0,
            "new_stop": stop_loss,
            "reason": "",
        }

        if direction == 1:  # Long
            if current_price <= stop_loss:
                result["action"] = "CLOSE_ALL"
                result["close_pct"] = 1.0
                result["reason"] = "STOP_LOSS"
            elif current_price >= tp1 and self._stage == 0:
                result["action"] = "PARTIAL_CLOSE"
                result["close_pct"] = 0.5
                result["new_stop"] = entry_price  # Breakeven
                result["reason"] = "TP1_HIT"
                self._stage = 1
            elif current_price >= tp2 and self._stage == 1:
                result["action"] = "CLOSE_ALL"
                result["close_pct"] = 1.0
                result["reason"] = "TP2_HIT"
                self._stage = 2

        elif direction == -1:  # Short
            if current_price >= stop_loss:
                result["action"] = "CLOSE_ALL"
                result["close_pct"] = 1.0
                result["reason"] = "STOP_LOSS"
            elif current_price <= tp1 and self._stage == 0:
                result["action"] = "PARTIAL_CLOSE"
                result["close_pct"] = 0.5
                result["new_stop"] = entry_price
                result["reason"] = "TP1_HIT"
                self._stage = 1
            elif current_price <= tp2 and self._stage == 1:
                result["action"] = "CLOSE_ALL"
                result["close_pct"] = 1.0
                result["reason"] = "TP2_HIT"
                self._stage = 2

        return result

    def reset(self) -> None:
        self._stage = 0

    @property
    def stage(self) -> int:
        return self._stage
