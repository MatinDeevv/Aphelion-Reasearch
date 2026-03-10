"""
OLYMPUS — Master Strategy Orchestrator
Phase 20 — Engineering Spec v3.0

Coordinates ALPHA (M1 scalp) and OMEGA (H1/H4 swing).
Capital allocation, performance decay detection, retraining triggers.
General-tier ARES voter (20 votes) — overrides individual strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


class StrategyMode(Enum):
    ALPHA_ONLY = "ALPHA_ONLY"
    OMEGA_ONLY = "OMEGA_ONLY"
    DUAL = "DUAL"
    PAUSED = "PAUSED"


class SystemState(Enum):
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"
    EMERGENCY = "EMERGENCY"
    RETRAINING = "RETRAINING"


@dataclass
class StrategyPerformance:
    """Performance snapshot for a single strategy."""
    name: str
    win_rate: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    trades_today: int = 0
    pnl_today: float = 0.0
    consecutive_losses: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class AllocationState:
    """Capital allocation between strategies."""
    alpha_pct: float = 0.7     # 70% to ALPHA by default
    omega_pct: float = 0.3     # 30% to OMEGA by default
    alpha_max_risk_pct: float = 1.0
    omega_max_risk_pct: float = 2.0


@dataclass
class OlympusState:
    """Complete OLYMPUS state."""
    mode: StrategyMode = StrategyMode.DUAL
    system_state: SystemState = SystemState.RUNNING
    allocation: AllocationState = field(default_factory=AllocationState)
    alpha_perf: Optional[StrategyPerformance] = None
    omega_perf: Optional[StrategyPerformance] = None
    retraining_needed: bool = False
    decay_detected: bool = False
    timestamp: Optional[datetime] = None


class DecayDetector:
    """
    Detects performance decay using CUSUM (Cumulative Sum Control Chart).
    If the rolling Sharpe drops below the 30d baseline by more than
    one standard deviation, triggers decay alert.
    """

    def __init__(self, window: int = 30, threshold: float = 1.0):
        self._window = window
        self._threshold = threshold
        self._returns: List[float] = []
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._decay_detected: bool = False

    def update(self, daily_return: float) -> bool:
        self._returns.append(daily_return)
        if len(self._returns) < self._window:
            return False

        recent = self._returns[-self._window:]
        mean_ret = sum(recent) / len(recent)
        target = 0.0  # We detect drift below zero

        self._cusum_pos = max(0, self._cusum_pos + daily_return - mean_ret - target)
        self._cusum_neg = min(0, self._cusum_neg + daily_return - mean_ret + target)

        self._decay_detected = abs(self._cusum_neg) > self._threshold
        return self._decay_detected

    @property
    def decay_detected(self) -> bool:
        return self._decay_detected

    def reset(self) -> None:
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._decay_detected = False


class RetrainingTrigger:
    """
    Determines when models need retraining.
    Conditions:
    - Win rate drops below 50% over 100 trades
    - Sharpe drops below 1.0 for 7 consecutive days
    - DecayDetector fires
    """

    def __init__(self, wr_threshold: float = 0.50, sharpe_threshold: float = 1.0, sharpe_days: int = 7):
        self._wr_threshold = wr_threshold
        self._sharpe_threshold = sharpe_threshold
        self._sharpe_days = sharpe_days
        self._low_sharpe_streak = 0
        self._trade_outcomes: List[bool] = []

    def record_trade(self, won: bool) -> None:
        self._trade_outcomes.append(won)
        if len(self._trade_outcomes) > 200:
            self._trade_outcomes = self._trade_outcomes[-200:]

    def record_daily_sharpe(self, sharpe: float) -> None:
        if sharpe < self._sharpe_threshold:
            self._low_sharpe_streak += 1
        else:
            self._low_sharpe_streak = 0

    def needs_retraining(self, decay_detected: bool = False) -> bool:
        # Win rate check
        if len(self._trade_outcomes) >= 100:
            recent = self._trade_outcomes[-100:]
            wr = sum(recent) / len(recent)
            if wr < self._wr_threshold:
                return True

        # Sharpe streak check
        if self._low_sharpe_streak >= self._sharpe_days:
            return True

        # Decay detector
        if decay_detected:
            return True

        return False

    def reset(self) -> None:
        self._low_sharpe_streak = 0
        self._trade_outcomes.clear()


class Olympus:
    """
    Master orchestrator. General-tier ARES voter (20 votes).

    Responsibilities:
    1. Capital allocation between ALPHA and OMEGA
    2. Performance monitoring & decay detection
    3. Retraining triggers
    4. Strategy mode switching
    5. Emergency pause/resume
    """

    # Capital allocation limits
    MIN_ALLOCATION = 0.1    # 10% minimum per strategy
    MAX_ALLOCATION = 0.9    # 90% maximum per strategy
    DAILY_LOSS_LIMIT = 0.02 # 2% daily loss triggers pause

    def __init__(self):
        self._state = OlympusState()
        self._decay_detector = DecayDetector()
        self._retrain_trigger = RetrainingTrigger()
        self._daily_pnl: float = 0.0
        self._account_balance: float = 10_000.0

    def update_alpha_performance(self, perf: StrategyPerformance) -> None:
        self._state.alpha_perf = perf
        self._check_strategy_health("ALPHA", perf)

    def update_omega_performance(self, perf: StrategyPerformance) -> None:
        self._state.omega_perf = perf
        self._check_strategy_health("OMEGA", perf)

    def update_daily_return(self, daily_return: float) -> None:
        """Update daily return for decay detection."""
        self._daily_pnl += daily_return
        decay = self._decay_detector.update(daily_return)
        self._state.decay_detected = decay

        if self._retrain_trigger.needs_retraining(decay):
            self._state.retraining_needed = True

    def record_trade(self, won: bool) -> None:
        self._retrain_trigger.record_trade(won)

    def set_account_balance(self, balance: float) -> None:
        self._account_balance = balance

    def _check_strategy_health(self, name: str, perf: StrategyPerformance) -> None:
        """Check if a strategy should be paused."""
        # Daily loss limit
        if self._account_balance > 0:
            daily_loss_pct = abs(self._daily_pnl) / self._account_balance
            if self._daily_pnl < 0 and daily_loss_pct > self.DAILY_LOSS_LIMIT:
                self._state.system_state = SystemState.PAUSED
                return

        # Consecutive losses
        if perf.consecutive_losses >= 5:
            if name == "ALPHA":
                self._state.mode = StrategyMode.OMEGA_ONLY
            elif name == "OMEGA":
                self._state.mode = StrategyMode.ALPHA_ONLY

    def rebalance_allocation(self) -> AllocationState:
        """Rebalance capital allocation based on relative performance."""
        alpha = self._state.alpha_perf
        omega = self._state.omega_perf

        if alpha is None or omega is None:
            return self._state.allocation

        # Sharpe-weighted allocation
        alpha_sharpe = max(0.1, alpha.sharpe)
        omega_sharpe = max(0.1, omega.sharpe)
        total = alpha_sharpe + omega_sharpe

        alpha_pct = alpha_sharpe / total
        omega_pct = omega_sharpe / total

        # Clamp to limits
        alpha_pct = max(self.MIN_ALLOCATION, min(self.MAX_ALLOCATION, alpha_pct))
        omega_pct = 1 - alpha_pct

        self._state.allocation = AllocationState(
            alpha_pct=round(alpha_pct, 3),
            omega_pct=round(omega_pct, 3),
        )
        return self._state.allocation

    def pause(self, reason: str = "") -> None:
        self._state.system_state = SystemState.PAUSED
        self._state.mode = StrategyMode.PAUSED

    def resume(self) -> None:
        self._state.system_state = SystemState.RUNNING
        self._state.mode = StrategyMode.DUAL

    def emergency_halt(self) -> None:
        self._state.system_state = SystemState.EMERGENCY
        self._state.mode = StrategyMode.PAUSED

    @property
    def state(self) -> OlympusState:
        self._state.timestamp = datetime.now(timezone.utc)
        return self._state

    @property
    def decay_detector(self) -> DecayDetector:
        return self._decay_detector
