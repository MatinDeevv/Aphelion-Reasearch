"""
NEMESIS — Stress Monitor
Phase 14 — Engineering Spec v3.0

Enhanced real-time monitoring of system-wide stress indicators.
Feeds data to the contrarian engine and NEMESIS sub-modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressSnapshot:
    """Point-in-time stress measurement."""
    timestamp: float
    composite_score: float
    win_rate_20: float
    regime_accuracy: float
    hydra_confidence_wr: float
    consecutive_losses: int
    failed_breakouts: int
    modules_failing: int


class EnhancedStressMonitor:
    """
    Advanced stress monitoring with history tracking and trend detection.
    Extends the base StressMonitor with time-series analysis.
    """

    def __init__(self, history_window: int = 500):
        self._history: List[StressSnapshot] = []
        self._window = history_window
        self._recent_outcomes: List[bool] = []
        self._high_conf_outcomes: List[bool] = []
        self._consecutive_losses: int = 0
        self._failed_breakouts: int = 0
        self._module_failures: Dict[str, int] = {}

    def record_trade(self, is_win: bool, was_high_conf: bool = False,
                     was_breakout: bool = False) -> None:
        """Record a trade outcome."""
        self._recent_outcomes.append(is_win)
        if len(self._recent_outcomes) > 100:
            self._recent_outcomes = self._recent_outcomes[-100:]

        if was_high_conf:
            self._high_conf_outcomes.append(is_win)
            if len(self._high_conf_outcomes) > 50:
                self._high_conf_outcomes = self._high_conf_outcomes[-50:]

        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if was_breakout:
                self._failed_breakouts += 1

    def record_module_failure(self, module_name: str) -> None:
        self._module_failures[module_name] = self._module_failures.get(module_name, 0) + 1

    @property
    def rolling_win_rate(self) -> float:
        if not self._recent_outcomes:
            return 0.5
        recent = self._recent_outcomes[-20:]
        return sum(recent) / len(recent)

    @property
    def high_conf_win_rate(self) -> float:
        if not self._high_conf_outcomes:
            return 0.6
        return sum(self._high_conf_outcomes) / len(self._high_conf_outcomes)

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def take_snapshot(self, regime_accuracy: float = 0.7) -> StressSnapshot:
        """Take a stress measurement."""
        snap = StressSnapshot(
            timestamp=time.time(),
            composite_score=self._compute_composite(),
            win_rate_20=self.rolling_win_rate,
            regime_accuracy=regime_accuracy,
            hydra_confidence_wr=self.high_conf_win_rate,
            consecutive_losses=self._consecutive_losses,
            failed_breakouts=self._failed_breakouts,
            modules_failing=sum(1 for v in self._module_failures.values() if v > 0),
        )
        self._history.append(snap)
        if len(self._history) > self._window:
            self._history = self._history[-self._window:]
        return snap

    def _compute_composite(self) -> float:
        """Composite stress score combining all factors."""
        score = 0.0
        wr = self.rolling_win_rate
        if wr < 0.45:
            score += 0.3 * (1.0 - wr / 0.45)
        if self._consecutive_losses >= 5:
            score += 0.2 * min(1.0, (self._consecutive_losses - 4) / 6)
        if self._failed_breakouts >= 3:
            score += 0.15
        hcwr = self.high_conf_win_rate
        if hcwr < 0.5:
            score += 0.2 * (1.0 - hcwr / 0.5)
        if self._module_failures:
            score += 0.15 * min(1.0, sum(self._module_failures.values()) / 5)
        return min(1.0, score)

    def is_stress_rising(self, lookback: int = 10) -> bool:
        """Detect if stress is trending upward."""
        if len(self._history) < lookback + 1:
            return False
        recent = [s.composite_score for s in self._history[-lookback:]]
        older = [s.composite_score for s in self._history[-(lookback * 2):-lookback]]
        if not older:
            return False
        return float(np.mean(recent)) > float(np.mean(older)) * 1.2

    def reset_session(self) -> None:
        self._failed_breakouts = 0
        self._module_failures.clear()
