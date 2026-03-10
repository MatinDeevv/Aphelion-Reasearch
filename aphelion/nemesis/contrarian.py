"""
NEMESIS — Contrarian Signal Generator
Phase 14 — Engineering Spec v3.0

Dedicated contrarian logic: when the system is failing,
generate signals that oppose the prevailing consensus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from aphelion.nemesis.detector import NEMESISDetector, NEMESISSignal, StressMonitor

logger = logging.getLogger(__name__)


@dataclass
class ContrarianConfig:
    """Configuration for contrarian signal generation."""
    # Activation thresholds
    min_stress_to_activate: float = 0.5
    # Confidence scaling
    max_contrarian_confidence: float = 0.8
    # Cool-off period (bars to wait after contrarian trade before re-signaling)
    cooloff_bars: int = 10
    # Maximum consecutive contrarian trades
    max_consecutive: int = 3


class ContrarianEngine:
    """
    Generates contrarian signals when system stress is elevated.
    This is NEMESIS's active trading component.
    """

    def __init__(
        self,
        detector: Optional[NEMESISDetector] = None,
        config: Optional[ContrarianConfig] = None,
    ):
        self._detector = detector or NEMESISDetector()
        self._config = config or ContrarianConfig()
        self._bars_since_contrarian: int = 999
        self._consecutive_contrarian: int = 0

    def evaluate(
        self,
        ares_consensus: int,
        rolling_win_rate: float,
        regime_accuracy: float = 0.7,
        high_conf_win_rate: float = 0.6,
        consecutive_losses: int = 0,
        failed_breakouts: int = 0,
    ) -> NEMESISSignal:
        """Evaluate whether to generate a contrarian signal."""
        self._bars_since_contrarian += 1

        # Generate base NEMESIS signal
        signal = self._detector.generate_signal(
            ares_consensus=ares_consensus,
            rolling_win_rate_20=rolling_win_rate,
            regime_accuracy=regime_accuracy,
            high_conf_win_rate=high_conf_win_rate,
            consecutive_losses=consecutive_losses,
            failed_breakouts=failed_breakouts,
        )

        # Apply cooloff
        if self._bars_since_contrarian < self._config.cooloff_bars:
            return NEMESISSignal(
                direction=0, confidence=0.0,
                reason="CONTRARIAN_COOLOFF",
                stress_score=signal.stress_score,
            )

        # Apply max consecutive limit
        if self._consecutive_contrarian >= self._config.max_consecutive:
            self._consecutive_contrarian = 0
            return NEMESISSignal(
                direction=0, confidence=0.0,
                reason="MAX_CONSECUTIVE_REACHED",
                stress_score=signal.stress_score,
            )

        # Check if stress is high enough to go contrarian
        if signal.stress_score < self._config.min_stress_to_activate:
            self._consecutive_contrarian = 0
            return signal

        # Scale confidence
        signal.confidence = min(
            signal.confidence,
            self._config.max_contrarian_confidence,
        )

        if signal.direction != 0:
            self._bars_since_contrarian = 0
            self._consecutive_contrarian += 1

        return signal

    def reset(self) -> None:
        self._bars_since_contrarian = 999
        self._consecutive_contrarian = 0
