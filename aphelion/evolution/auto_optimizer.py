"""
APHELION AutoOptimizer — Continuous Improvement Engine
Phase 16 — Engineering Spec v3.0

Background service that continuously evaluates system performance
and schedules optimization runs when degradation is detected.

Coordinates:
- FORGE parameter optimization
- PROMETHEUS genome evolution
- HYDRA model retraining
- Feature importance re-evaluation

Decision loop:
1. Monitor rolling performance via KRONOS
2. If degradation detected → schedule optimization
3. Run optimization → validate via TITAN
4. If validated → hot-swap parameters
5. Log results to KRONOS journal
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OptAction(Enum):
    """Types of optimization actions."""
    FORGE_PARAM_OPT = auto()
    PROMETHEUS_EVOLVE = auto()
    HYDRA_RETRAIN = auto()
    FEATURE_REEVAL = auto()
    FULL_REOPTIMIZE = auto()


@dataclass
class DegradationSignal:
    """Signal indicating performance degradation."""
    metric: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_severe(self) -> bool:
        return self.degradation_pct > 0.20


@dataclass
class OptimizationRun:
    """Record of an optimization run."""
    run_id: str
    action: OptAction
    trigger: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: dict = field(default_factory=dict)
    titan_approved: bool = False
    applied: bool = False


class PerformanceMonitor:
    """Monitors system performance and detects degradation."""

    def __init__(
        self,
        sharpe_threshold: float = 1.0,
        win_rate_threshold: float = 0.50,
        drawdown_threshold: float = 0.08,
        lookback_trades: int = 100,
    ):
        self._sharpe_thresh = sharpe_threshold
        self._wr_thresh = win_rate_threshold
        self._dd_thresh = drawdown_threshold
        self._lookback = lookback_trades
        self._baseline_sharpe: float = 0.0
        self._baseline_wr: float = 0.0

    def set_baseline(self, sharpe: float, win_rate: float) -> None:
        """Set performance baseline (after TITAN approval)."""
        self._baseline_sharpe = sharpe
        self._baseline_wr = win_rate

    def check_degradation(
        self,
        current_sharpe: float,
        current_win_rate: float,
        current_drawdown: float,
    ) -> List[DegradationSignal]:
        """Check for performance degradation."""
        signals = []

        # Sharpe drop
        if self._baseline_sharpe > 0:
            drop = (self._baseline_sharpe - current_sharpe) / self._baseline_sharpe
            if drop > 0.15:
                signals.append(DegradationSignal(
                    metric="sharpe_ratio",
                    current_value=current_sharpe,
                    baseline_value=self._baseline_sharpe,
                    degradation_pct=drop,
                ))

        # Win rate drop
        if self._baseline_wr > 0:
            drop = (self._baseline_wr - current_win_rate) / self._baseline_wr
            if drop > 0.10:
                signals.append(DegradationSignal(
                    metric="win_rate",
                    current_value=current_win_rate,
                    baseline_value=self._baseline_wr,
                    degradation_pct=drop,
                ))

        # Drawdown breach
        if current_drawdown > self._dd_thresh:
            signals.append(DegradationSignal(
                metric="drawdown",
                current_value=current_drawdown,
                baseline_value=self._dd_thresh,
                degradation_pct=current_drawdown / self._dd_thresh - 1,
            ))

        return signals


class AutoOptimizer:
    """
    Continuous improvement engine.

    Monitors performance → detects degradation → schedules optimization →
    validates via TITAN → hot-swaps if approved.
    """

    def __init__(self):
        self._monitor = PerformanceMonitor()
        self._runs: List[OptimizationRun] = []
        self._cooldown_hours: float = 24.0
        self._last_optimization: Optional[datetime] = None
        self._callbacks: Dict[OptAction, Callable] = {}
        self._enabled: bool = True

    def set_baseline(self, sharpe: float, win_rate: float) -> None:
        """Set performance baseline after TITAN approval."""
        self._monitor.set_baseline(sharpe, win_rate)

    def register_callback(self, action: OptAction, callback: Callable) -> None:
        """Register an optimization callback for an action type."""
        self._callbacks[action] = callback

    def evaluate(
        self,
        current_sharpe: float,
        current_win_rate: float,
        current_drawdown: float,
    ) -> List[DegradationSignal]:
        """Evaluate current performance and return any degradation signals."""
        if not self._enabled:
            return []
        return self._monitor.check_degradation(
            current_sharpe, current_win_rate, current_drawdown
        )

    def should_optimize(self, signals: List[DegradationSignal]) -> Optional[OptAction]:
        """Determine which optimization to run based on signals."""
        if not signals:
            return None

        # Check cooldown
        if self._last_optimization is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_optimization).total_seconds() / 3600
            if elapsed < self._cooldown_hours:
                return None

        # Determine action based on signal severity
        severe = any(s.is_severe for s in signals)

        if severe:
            return OptAction.FULL_REOPTIMIZE

        # Check which metric is degrading
        metrics = {s.metric for s in signals}
        if "sharpe_ratio" in metrics:
            return OptAction.FORGE_PARAM_OPT
        if "win_rate" in metrics:
            return OptAction.HYDRA_RETRAIN
        return OptAction.FEATURE_REEVAL

    def run_optimization(self, action: OptAction, trigger: str = "") -> OptimizationRun:
        """Execute an optimization run."""
        run = OptimizationRun(
            run_id=f"opt_{int(time.time())}",
            action=action,
            trigger=trigger,
            start_time=datetime.now(timezone.utc),
        )

        callback = self._callbacks.get(action)
        if callback is not None:
            try:
                result = callback()
                run.result = result if isinstance(result, dict) else {"status": "completed"}
            except Exception as e:
                logger.error("Optimization %s failed: %s", action.name, e)
                run.result = {"status": "failed", "error": str(e)}
        else:
            run.result = {"status": "no_callback"}

        run.end_time = datetime.now(timezone.utc)
        self._runs.append(run)
        self._last_optimization = datetime.now(timezone.utc)
        return run

    def approve_and_apply(self, run: OptimizationRun) -> bool:
        """Mark a run as TITAN-approved and applied."""
        run.titan_approved = True
        run.applied = True
        logger.info("AutoOptimizer: Applied optimization %s (%s)", run.run_id, run.action.name)
        return True

    @property
    def total_runs(self) -> int:
        return len(self._runs)

    @property
    def successful_runs(self) -> int:
        return sum(1 for r in self._runs if r.applied)

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def get_run_history(self, last_n: int = 20) -> List[OptimizationRun]:
        return self._runs[-last_n:]
