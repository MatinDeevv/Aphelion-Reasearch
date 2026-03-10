"""
FLOW Analyzer — Main coordinator for all FLOW sub-modules.
Produces a unified FlowSignal for ARES voting.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from aphelion.flow.liquidity import LiquidityZoneDetector, LiquidityZone
from aphelion.flow.orderflow import OrderFlowAnalyzer
from aphelion.flow.imbalance import ImbalanceTracker
from aphelion.flow.absorption import AbsorptionDetector
from aphelion.flow.sweep_detector import StopHuntDetector


@dataclass
class FlowSignal:
    """FLOW → ARES output."""
    direction: int           # 1=BUY, -1=SELL, 0=FLAT
    confidence: float        # [0, 1]

    # Sub-signals
    delta: float             # Volume delta [-1, 1]
    near_liquidity_zone: bool
    zone_distance_pips: float
    stop_hunt_detected: bool
    absorption_detected: bool

    # Context
    session: str = ""
    volatility_regime: str = ""


class FlowAnalyzer:
    """
    Main FLOW coordinator. Aggregates liquidity, order flow, imbalance,
    absorption, and sweep detection into a single FlowSignal.

    Commander-tier ARES voter (10 votes).
    """

    def __init__(self):
        self._liquidity = LiquidityZoneDetector()
        self._orderflow = OrderFlowAnalyzer()
        self._imbalance = ImbalanceTracker()
        self._absorption = AbsorptionDetector()
        self._sweep = StopHuntDetector()

    def analyze(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        opens: Optional[np.ndarray] = None,
        session: str = "",
        volatility_regime: str = "",
    ) -> FlowSignal:
        """Run full FLOW analysis on recent bar data."""
        if len(closes) < 5:
            return FlowSignal(
                direction=0, confidence=0.0, delta=0.0,
                near_liquidity_zone=False, zone_distance_pips=999.0,
                stop_hunt_detected=False, absorption_detected=False,
                session=session, volatility_regime=volatility_regime,
            )

        # 1. Detect liquidity zones
        zones = self._liquidity.detect_zones(highs, lows, closes, volumes)

        # 2. Compute order flow delta from bar data
        of_state = self._orderflow.compute_bar_delta(closes, volumes)

        # 3. Check imbalance
        imb_state = self._imbalance.update(
            of_state.buy_volume, of_state.sell_volume
        )

        # 4. Check absorption
        if opens is not None and len(opens) > 0:
            absorption = self._absorption.update(
                float(highs[-1]), float(lows[-1]), float(closes[-1]),
                float(volumes[-1]), float(opens[-1])
            )
        else:
            absorption = None

        # 5. Check stop hunt
        sweep = self._sweep.detect(highs, lows, closes, volumes, zones)

        # Compute distance to nearest zone
        current_price = float(closes[-1])
        zone_dist = 999.0
        if zones:
            zone_dist = min(abs(current_price - z.price) for z in zones)

        near_zone = zone_dist < 10.0  # Within 10 pips

        # Aggregate into direction/confidence
        direction, confidence = self._aggregate_signals(
            of_state.delta, imb_state.imbalance, sweep, absorption, near_zone
        )

        return FlowSignal(
            direction=direction,
            confidence=confidence,
            delta=of_state.delta,
            near_liquidity_zone=near_zone,
            zone_distance_pips=zone_dist,
            stop_hunt_detected=sweep is not None,
            absorption_detected=absorption is not None,
            session=session,
            volatility_regime=volatility_regime,
        )

    @staticmethod
    def _aggregate_signals(
        delta: float,
        imbalance: float,
        sweep,
        absorption,
        near_zone: bool,
    ) -> tuple:
        """Combine FLOW sub-signals into direction + confidence."""
        score = 0.0

        # Delta contribution (strongest signal)
        score += delta * 0.35

        # Imbalance contribution
        score += imbalance * 0.25

        # Stop hunt detection (very high quality signal)
        if sweep is not None:
            score += sweep.direction * sweep.confidence * 0.30

        # Absorption (moderate signal)
        if absorption is not None:
            # Absorption of selling = bullish, absorption of buying = bearish
            abs_dir = 1 if absorption.direction == "SELL_ABSORBED" else -1
            score += abs_dir * absorption.strength * 0.10

        # Direction
        if abs(score) < 0.15:
            direction = 0
            confidence = abs(score)
        else:
            direction = 1 if score > 0 else -1
            confidence = min(1.0, abs(score))

        # Boost confidence if near liquidity zone
        if near_zone and direction != 0:
            confidence = min(1.0, confidence * 1.2)

        return direction, confidence

    def reset(self) -> None:
        self._orderflow.reset()
        self._imbalance.reset()
        self._absorption.reset()
