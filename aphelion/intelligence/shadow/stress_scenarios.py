"""
SHADOW — Stress Scenarios
Phase 13 — Engineering Spec v3.0

Predefined stress test scenarios for validating system resilience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from aphelion.intelligence.shadow.regime_simulator import (
    AdvancedRegimeSimulator,
    RegimeScenario,
    SyntheticBar,
)


@dataclass
class StressTestResult:
    """Result of running a stress test scenario."""
    scenario_name: str
    n_bars: int
    max_drawdown_pct: float
    final_pnl: float
    trades_taken: int
    l1_triggers: int
    l2_triggers: int
    l3_triggers: int
    passed: bool
    notes: str = ""


class StressScenarioLibrary:
    """
    Predefined stress test scenarios per engineering spec:
    - Flash crash (XAU drops $80 in 5 minutes)
    - Liquidity void (spread widens 5x, volume drops 90%)
    - Regime whipsaw (rapid trending→ranging→trending switches)
    - News spike (NFP/FOMC: 3σ move in 1 bar)
    - Weekend gap (Monday open gap of 2%)
    """

    SCENARIOS: Dict[str, List[RegimeScenario]] = {
        "flash_crash": [
            RegimeScenario("pre_crash", "TRENDING_BULL", 60, drift=0.0002),
            RegimeScenario("crash", "CRISIS", 30),
            RegimeScenario("recovery", "VOLATILE", 60),
        ],
        "liquidity_void": [
            RegimeScenario("normal", "RANGING", 100),
            RegimeScenario("void", "VOLATILE", 50, volatility=0.005),
            RegimeScenario("recovery", "RANGING", 100),
        ],
        "regime_whipsaw": [
            RegimeScenario("trend1", "TRENDING_BULL", 30, drift=0.001),
            RegimeScenario("range1", "RANGING", 20),
            RegimeScenario("trend2", "TRENDING_BEAR", 30, drift=0.001),
            RegimeScenario("range2", "RANGING", 20),
            RegimeScenario("trend3", "TRENDING_BULL", 30, drift=0.0008),
        ],
        "news_spike": [
            RegimeScenario("pre_news", "RANGING", 60, volatility=0.0005),
            RegimeScenario("spike", "VOLATILE", 5, volatility=0.015),
            RegimeScenario("post_news", "VOLATILE", 30, volatility=0.003),
            RegimeScenario("settle", "RANGING", 60),
        ],
        "weekend_gap": [
            RegimeScenario("friday", "RANGING", 60),
            RegimeScenario("gap", "VOLATILE", 3, volatility=0.02),
            RegimeScenario("monday", "VOLATILE", 30, volatility=0.003),
            RegimeScenario("settle", "RANGING", 60),
        ],
    }

    def __init__(self):
        self._simulator = AdvancedRegimeSimulator()

    def generate_scenario(self, name: str) -> List[SyntheticBar]:
        """Generate synthetic bars for a named stress scenario."""
        phases = self.SCENARIOS.get(name)
        if phases is None:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(self.SCENARIOS.keys())}")
        return self._simulator.generate_transition(phases)

    def list_scenarios(self) -> List[str]:
        return list(self.SCENARIOS.keys())

    def generate_all(self) -> Dict[str, List[SyntheticBar]]:
        """Generate bars for all scenarios."""
        return {name: self.generate_scenario(name) for name in self.SCENARIOS}
