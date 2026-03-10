"""
NEMESIS — Anti-Regime Contrarian Detector
Phase 14 — Engineering Spec v3.0

Detects when the current strategy is failing and votes AGAINST
the system's own consensus. APHELION's built-in contrarian.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NEMESISSignal:
    direction: int           # Opposite of consensus when stressed
    confidence: float
    reason: str
    stress_score: float = 0.0


class NEMESISDetector:
    """
    NEMESIS votes AGAINST the prevailing consensus when:
    1. Recent win rate has dropped below 45% (trailing 20 trades)
    2. Current regime classification accuracy < 60%
    3. HYDRA confidence is high but recent high-conf trades are losers
    4. Multiple failed breakout attempts detected

    Lieutenant-tier ARES voter (5 votes).
    """

    def __init__(
        self,
        win_rate_threshold: float = 0.45,
        regime_accuracy_threshold: float = 0.60,
        high_stress_threshold: float = 0.7,
        moderate_stress_threshold: float = 0.5,
    ):
        self._wr_thresh = win_rate_threshold
        self._regime_thresh = regime_accuracy_threshold
        self._high_stress = high_stress_threshold
        self._mod_stress = moderate_stress_threshold

    def generate_signal(
        self,
        ares_consensus: int,
        rolling_win_rate_20: float,
        regime_accuracy: float = 0.7,
        high_conf_win_rate: float = 0.6,
        consecutive_losses: int = 0,
        failed_breakouts: int = 0,
    ) -> NEMESISSignal:
        """Generate NEMESIS signal based on system stress indicators."""
        stress_score = self._compute_stress_score(
            rolling_win_rate_20,
            regime_accuracy,
            high_conf_win_rate,
            consecutive_losses,
            failed_breakouts,
        )

        if stress_score > self._high_stress:
            return NEMESISSignal(
                direction=-ares_consensus if ares_consensus != 0 else 0,
                confidence=min(1.0, stress_score),
                reason="HIGH_SYSTEM_STRESS",
                stress_score=stress_score,
            )
        elif stress_score > self._mod_stress:
            return NEMESISSignal(
                direction=0,
                confidence=0.5,
                reason="MODERATE_STRESS",
                stress_score=stress_score,
            )
        else:
            return NEMESISSignal(
                direction=0,
                confidence=0.0,
                reason="SYSTEM_HEALTHY",
                stress_score=stress_score,
            )

    def _compute_stress_score(
        self,
        win_rate: float,
        regime_accuracy: float,
        high_conf_wr: float,
        consec_losses: int,
        failed_breakouts: int,
    ) -> float:
        """Compute composite stress score [0, 1]."""
        score = 0.0

        # Win rate below threshold
        if win_rate < self._wr_thresh:
            score += 0.3 * (1 - win_rate / self._wr_thresh)

        # Regime accuracy below threshold
        if regime_accuracy < self._regime_thresh:
            score += 0.2 * (1 - regime_accuracy / self._regime_thresh)

        # High-confidence trades losing
        if high_conf_wr < 0.5:
            score += 0.25 * (1 - high_conf_wr / 0.5)

        # Consecutive losses
        if consec_losses >= 5:
            score += 0.15 * min(1.0, (consec_losses - 4) / 6)

        # Failed breakouts in session
        if failed_breakouts >= 3:
            score += 0.1 * min(1.0, (failed_breakouts - 2) / 5)

        return min(1.0, score)


class StressMonitor:
    """Real-time system stress tracking for NEMESIS."""

    def __init__(self):
        self._failed_breakouts = 0
        self._session_trades = 0
        self._session_losses = 0

    def record_trade_outcome(self, is_win: bool, was_breakout: bool = False) -> None:
        self._session_trades += 1
        if not is_win:
            self._session_losses += 1
            if was_breakout:
                self._failed_breakouts += 1

    @property
    def failed_breakouts(self) -> int:
        return self._failed_breakouts

    @property
    def session_win_rate(self) -> float:
        if self._session_trades == 0:
            return 0.5
        return 1 - (self._session_losses / self._session_trades)

    def reset_session(self) -> None:
        self._failed_breakouts = 0
        self._session_trades = 0
        self._session_losses = 0
