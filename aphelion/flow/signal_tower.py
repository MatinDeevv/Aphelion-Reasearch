"""
SIGNAL TOWER — Independent Technical Analysis Voters
Phase 18 — Engineering Spec v3.0

Collection of independent TA voters, each operating as a standalone
Commander-tier ARES voter. Critical: they NEVER filter or communicate
with each other — pure independent signals.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Vote:
    direction: int       # 1=BUY, -1=SELL, 0=FLAT
    confidence: float    # [0, 1]
    voter_name: str = ""


class HalfTrendVoter:
    """HalfTrend trend direction. Commander-tier (10 votes)."""

    def __init__(self, amplitude: int = 2, atr_period: int = 100):
        self._amplitude = amplitude
        self._atr_period = atr_period

    def vote(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Vote:
        if len(closes) < max(self._amplitude, self._atr_period) + 1:
            return Vote(0, 0.0, "HalfTrend")

        atr = self._compute_atr(highs, lows, closes, self._atr_period)
        if atr <= 0:
            return Vote(0, 0.0, "HalfTrend")

        recent_high = float(np.max(highs[-self._amplitude:]))
        recent_low = float(np.min(lows[-self._amplitude:]))
        upper = recent_high - atr * 1.5
        lower = recent_low + atr * 1.5
        current = float(closes[-1])

        if current > upper:
            strength = min(1.0, abs(current - upper) / atr)
            return Vote(1, strength, "HalfTrend")
        elif current < lower:
            strength = min(1.0, abs(current - lower) / atr)
            return Vote(-1, strength, "HalfTrend")
        return Vote(0, 0.3, "HalfTrend")

    @staticmethod
    def _compute_atr(highs, lows, closes, period):
        if len(closes) < 2:
            return 0.0
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(
            np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))


class EMAStackVoter:
    """EMA 8/21/50 stack alignment. Commander-tier (10 votes)."""

    def vote(self, closes: np.ndarray) -> Vote:
        if len(closes) < 50:
            return Vote(0, 0.0, "EMAStack")

        ema8 = self._ema(closes, 8)
        ema21 = self._ema(closes, 21)
        ema50 = self._ema(closes, 50)

        if ema8 > ema21 > ema50:
            spread = ema8 - ema50
            confidence = min(1.0, spread / (ema50 * 0.001)) if ema50 > 0 else 0.5
            return Vote(1, confidence, "EMAStack")
        elif ema8 < ema21 < ema50:
            spread = ema50 - ema8
            confidence = min(1.0, spread / (ema50 * 0.001)) if ema50 > 0 else 0.5
            return Vote(-1, confidence, "EMAStack")
        return Vote(0, 0.3, "EMAStack")

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        mult = 2.0 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * mult + ema
        return float(ema)


class VWAPPositionVoter:
    """Price position relative to VWAP. Commander-tier (10 votes)."""

    def vote(self, close: float, vwap: float) -> Vote:
        if vwap <= 0:
            return Vote(0, 0.0, "VWAPPosition")

        distance_pct = (close - vwap) / vwap * 100

        if abs(distance_pct) < 0.1:
            return Vote(0, 0.2, "VWAPPosition")

        direction = 1 if distance_pct > 0 else -1
        confidence = min(1.0, abs(distance_pct) / 0.5)
        return Vote(direction, confidence, "VWAPPosition")


class BreakoutDetector:
    """Range breakout with volume confirmation. Commander-tier (10 votes)."""

    def vote(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        volumes: np.ndarray,
    ) -> Vote:
        if len(closes) < 30:
            return Vote(0, 0.0, "Breakout")

        # Compute ATR for range definition
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(
            np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr_20 = float(np.mean(tr[-20:])) if len(tr) >= 20 else float(np.mean(tr))
        atr_50pct = float(np.percentile(tr[-50:], 50)) if len(tr) >= 50 else atr_20

        # Is market in consolidation? (ATR below median)
        in_range = atr_20 < atr_50pct

        if not in_range:
            return Vote(0, 0.2, "Breakout")

        # Range high/low from last 20 bars
        range_high = float(np.max(highs[-20:]))
        range_low = float(np.min(lows[-20:]))
        current = float(closes[-1])

        # Volume confirmation
        avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        vol_confirmed = float(volumes[-1]) > avg_vol * 1.5

        # Breakout above
        if current > range_high + atr_20 and vol_confirmed:
            return Vote(1, 0.75, "Breakout")
        # Breakout below
        elif current < range_low - atr_20 and vol_confirmed:
            return Vote(-1, 0.75, "Breakout")

        return Vote(0, 0.1, "Breakout")


class RSIExtremeVoter:
    """RSI overbought/oversold readings. Commander-tier (10 votes)."""

    def vote(self, rsi: float) -> Vote:
        if rsi > 80:
            return Vote(-1, 0.7, "RSIExtreme")
        elif rsi > 70:
            return Vote(-1, 0.4, "RSIExtreme")
        elif rsi < 20:
            return Vote(1, 0.7, "RSIExtreme")
        elif rsi < 30:
            return Vote(1, 0.4, "RSIExtreme")
        return Vote(0, 0.1, "RSIExtreme")


class StructureVoter:
    """Market structure (HH/HL vs LH/LL). Commander-tier (10 votes)."""

    def vote(self, highs: np.ndarray, lows: np.ndarray) -> Vote:
        if len(highs) < 10:
            return Vote(0, 0.0, "Structure")

        swing_highs = []
        swing_lows = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(float(highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(float(lows[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return Vote(0, 0.2, "Structure")

        hh = swing_highs[-1] > swing_highs[-2]
        hl = swing_lows[-1] > swing_lows[-2]
        lh = swing_highs[-1] < swing_highs[-2]
        ll = swing_lows[-1] < swing_lows[-2]

        if hh and hl:
            return Vote(1, 0.7, "Structure")
        elif lh and ll:
            return Vote(-1, 0.7, "Structure")
        return Vote(0, 0.3, "Structure")


class SessionMomentumVoter:
    """Session-open momentum direction. Commander-tier (10 votes)."""

    def vote(self, closes: np.ndarray, session_open_idx: int = 0) -> Vote:
        """Vote based on momentum since session open."""
        if len(closes) < 5 or session_open_idx >= len(closes):
            return Vote(0, 0.0, "SessionMomentum")

        session_open = float(closes[session_open_idx])
        current = float(closes[-1])

        if session_open <= 0:
            return Vote(0, 0.0, "SessionMomentum")

        momentum_pct = (current - session_open) / session_open * 100

        if abs(momentum_pct) < 0.05:
            return Vote(0, 0.1, "SessionMomentum")

        direction = 1 if momentum_pct > 0 else -1
        confidence = min(1.0, abs(momentum_pct) / 0.3)
        return Vote(direction, confidence, "SessionMomentum")


class SignalTower:
    """
    Aggregator: feeds each voter to ARES independently.
    Each voter is a pure, independent signal source.
    """

    def __init__(self):
        self.halftrend = HalfTrendVoter()
        self.ema_stack = EMAStackVoter()
        self.vwap_position = VWAPPositionVoter()
        self.breakout = BreakoutDetector()
        self.rsi_extreme = RSIExtremeVoter()
        self.structure = StructureVoter()
        self.session_momentum = SessionMomentumVoter()

    def collect_votes(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        vwap: float = 0.0,
        rsi: float = 50.0,
        session_open_idx: int = 0,
    ) -> Dict[str, Vote]:
        """Collect all independent votes."""
        votes = {}
        votes["HalfTrend"] = self.halftrend.vote(highs, lows, closes)
        votes["EMAStack"] = self.ema_stack.vote(closes)
        votes["VWAPPosition"] = self.vwap_position.vote(float(closes[-1]), vwap)
        votes["Breakout"] = self.breakout.vote(highs, lows, closes, volumes)
        votes["RSIExtreme"] = self.rsi_extreme.vote(rsi)
        votes["Structure"] = self.structure.vote(highs, lows)
        votes["SessionMomentum"] = self.session_momentum.vote(closes, session_open_idx)
        return votes

    def get_aggregate(self, votes: Dict[str, Vote]) -> Vote:
        """Simple aggregation (each voter equal weight within SIGNAL TOWER)."""
        if not votes:
            return Vote(0, 0.0, "SignalTower")

        total_score = 0.0
        total_conf = 0.0
        for vote in votes.values():
            total_score += vote.direction * vote.confidence
            total_conf += vote.confidence

        n = len(votes)
        avg_score = total_score / n
        avg_conf = total_conf / n

        if abs(avg_score) < 0.15:
            return Vote(0, avg_conf, "SignalTower")

        direction = 1 if avg_score > 0 else -1
        return Vote(direction, min(1.0, abs(avg_score)), "SignalTower")
