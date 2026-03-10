"""
ORACLE — Macro Prediction Engine Core
Phase 19 — Engineering Spec v3.0

Generates probabilistic macro forecasts for gold direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Forecast:
    """Probabilistic forecast for gold direction."""
    direction: int          # -1, 0, 1
    probability_up: float   # P(price goes up)
    probability_down: float # P(price goes down)
    confidence: float       # [0, 1]
    horizon_bars: int       # Forecast horizon
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def edge(self) -> float:
        """Edge = |P(up) - P(down)|."""
        return abs(self.probability_up - self.probability_down)


class OracleCore:
    """
    Simple historical-frequency-based forecaster.
    Tracks regime-conditional returns to estimate direction probabilities.
    """

    def __init__(self, lookback: int = 200):
        self._lookback = lookback
        self._returns: List[float] = []
        self._regime_returns: Dict[str, List[float]] = {}

    def update(self, bar_return: float, regime: str = "UNKNOWN") -> None:
        """Record a bar return."""
        self._returns.append(bar_return)
        self._regime_returns.setdefault(regime, []).append(bar_return)

    def forecast(self, regime: str = "UNKNOWN", horizon_bars: int = 10) -> Forecast:
        """Generate a directional forecast."""
        # Use regime-specific if available, else all returns
        rets = self._regime_returns.get(regime, self._returns)
        if not rets:
            return Forecast(0, 0.5, 0.5, 0.0, horizon_bars)

        recent = rets[-self._lookback:]
        arr = np.array(recent)
        p_up = float(np.mean(arr > 0))
        p_down = float(np.mean(arr < 0))

        direction = 1 if p_up > 0.55 else (-1 if p_down > 0.55 else 0)
        confidence = abs(p_up - p_down)

        return Forecast(
            direction=direction,
            probability_up=round(p_up, 3),
            probability_down=round(p_down, 3),
            confidence=round(confidence, 3),
            horizon_bars=horizon_bars,
        )

    @property
    def total_observations(self) -> int:
        return len(self._returns)
