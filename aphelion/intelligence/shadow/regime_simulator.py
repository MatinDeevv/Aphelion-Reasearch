"""
SHADOW — Regime Simulator
Phase 13 — Engineering Spec v3.0

Generates synthetic market data for specific regime conditions.
Wraps and extends the base SyntheticBar generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from aphelion.intelligence.shadow.generator import RegimeSimulator, SyntheticBar

logger = logging.getLogger(__name__)


@dataclass
class RegimeScenario:
    """Configuration for a synthetic regime scenario."""
    name: str
    regime: str
    n_bars: int
    direction: int = 1
    volatility: float = 0.001
    drift: float = 0.0005
    description: str = ""


class AdvancedRegimeSimulator:
    """
    Extended regime simulator that generates complex multi-phase scenarios.
    Supports transition sequences, shock events, and calendar effects.
    """

    def __init__(self):
        self._base = RegimeSimulator()

    def generate_scenario(self, scenario: RegimeScenario) -> List[SyntheticBar]:
        """Generate bars for a specific regime scenario."""
        if scenario.regime in ("TRENDING_BULL", "TRENDING_BEAR"):
            direction = 1 if "BULL" in scenario.regime else -1
            return self._base.generate_trending(
                n_bars=scenario.n_bars,
                direction=direction,
                volatility=scenario.volatility,
                drift=scenario.drift,
            )
        elif scenario.regime == "RANGING":
            return self._generate_ranging(scenario.n_bars, scenario.volatility)
        elif scenario.regime == "VOLATILE":
            return self._generate_volatile(scenario.n_bars, scenario.volatility * 3)
        elif scenario.regime == "CRISIS":
            return self._generate_crisis(scenario.n_bars)
        else:
            return self._base.generate_trending(
                n_bars=scenario.n_bars,
                direction=scenario.direction,
                volatility=scenario.volatility,
            )

    def generate_transition(
        self,
        phases: List[RegimeScenario],
        transition_bars: int = 10,
    ) -> List[SyntheticBar]:
        """Generate a multi-phase scenario with transitions between regimes."""
        all_bars: List[SyntheticBar] = []
        for i, phase in enumerate(phases):
            phase_bars = self.generate_scenario(phase)
            all_bars.extend(phase_bars)
            # Insert transition noise between phases
            if i < len(phases) - 1 and transition_bars > 0:
                last_price = phase_bars[-1].close if phase_bars else 3000.0
                trans = self._generate_ranging(transition_bars, 0.002, start_price=last_price)
                all_bars.extend(trans)

        # Re-index timestamps
        for idx, bar in enumerate(all_bars):
            bar.timestamp_idx = idx
        return all_bars

    def _generate_ranging(self, n_bars: int, volatility: float,
                          start_price: float = 3000.0) -> List[SyntheticBar]:
        bars = []
        price = start_price
        for i in range(n_bars):
            ret = np.random.normal(0, volatility)
            new_price = price * (1 + ret)
            high = max(price, new_price) * (1 + abs(np.random.normal(0, volatility * 0.3)))
            low = min(price, new_price) * (1 - abs(np.random.normal(0, volatility * 0.3)))
            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=max(100, np.random.normal(3000, 500)),
                regime="RANGING",
            ))
            price = new_price
        return bars

    def _generate_volatile(self, n_bars: int, volatility: float) -> List[SyntheticBar]:
        bars = []
        price = 3000.0
        for i in range(n_bars):
            ret = np.random.normal(0, volatility) + np.random.choice([-1, 1]) * volatility * 0.5
            new_price = price * (1 + ret)
            high = max(price, new_price) * (1 + abs(np.random.normal(0, volatility)))
            low = min(price, new_price) * (1 - abs(np.random.normal(0, volatility)))
            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=max(100, np.random.normal(8000, 2000)),
                regime="VOLATILE",
            ))
            price = new_price
        return bars

    def _generate_crisis(self, n_bars: int) -> List[SyntheticBar]:
        """Flash crash + recovery scenario."""
        bars = []
        price = 3000.0
        crash_point = n_bars // 3
        recovery_point = 2 * n_bars // 3

        for i in range(n_bars):
            if i < crash_point:
                ret = np.random.normal(-0.003, 0.004)
            elif i < recovery_point:
                ret = np.random.normal(-0.008, 0.012)
            else:
                ret = np.random.normal(0.002, 0.005)

            new_price = price * (1 + ret)
            high = max(price, new_price) * 1.002
            low = min(price, new_price) * 0.998
            bars.append(SyntheticBar(
                timestamp_idx=i, open=price, high=high, low=low,
                close=new_price, volume=max(100, np.random.normal(15000, 5000)),
                regime="CRISIS",
            ))
            price = new_price
        return bars
