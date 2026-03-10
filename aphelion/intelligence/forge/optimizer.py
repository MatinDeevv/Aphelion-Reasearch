"""
FORGE — Strategy Parameter Optimizer
Phase 13 — Engineering Spec v3.0

Uses Bayesian optimization to tune strategy parameters.
Runs every 2 weeks: 200 trials × 100-bar backtest per trial.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import math


@dataclass
class ParameterSpec:
    """Definition of an optimizable parameter."""
    name: str
    min_val: float
    max_val: float
    current_val: float
    step: float = 0.01
    dtype: str = "continuous"  # "continuous" or "discrete"


@dataclass
class TrialResult:
    """Result of a single optimization trial."""
    trial_id: int
    parameters: Dict[str, float]
    fitness: float
    sharpe: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BayesianOptimizer:
    """
    Simple Bayesian-inspired optimizer using random exploration + exploitation.
    For full prod, wraps Optuna or scikit-optimize.
    """

    def __init__(self, parameter_space: List[ParameterSpec], n_initial: int = 20):
        self._space = {p.name: p for p in parameter_space}
        self._n_initial = n_initial
        self._trials: List[TrialResult] = []
        self._best_trial: Optional[TrialResult] = None

    def suggest(self) -> Dict[str, float]:
        """Suggest a new parameter set to try."""
        if len(self._trials) < self._n_initial:
            # Exploration: random sampling
            return self._random_sample()
        else:
            # Exploitation: perturb best known parameters
            return self._perturb_best()

    def _random_sample(self) -> Dict[str, float]:
        params = {}
        for name, spec in self._space.items():
            if spec.dtype == "discrete":
                params[name] = round(random.uniform(spec.min_val, spec.max_val) / spec.step) * spec.step
            else:
                params[name] = random.uniform(spec.min_val, spec.max_val)
        return params

    def _perturb_best(self) -> Dict[str, float]:
        if self._best_trial is None:
            return self._random_sample()

        params = dict(self._best_trial.parameters)
        for name, spec in self._space.items():
            # Perturb by ±10% of range
            range_size = spec.max_val - spec.min_val
            perturbation = random.gauss(0, range_size * 0.1)
            new_val = params.get(name, spec.current_val) + perturbation
            params[name] = max(spec.min_val, min(spec.max_val, new_val))
        return params

    def report(self, trial_id: int, parameters: Dict[str, float], fitness: float,
               sharpe: float = 0.0, win_rate: float = 0.0, max_drawdown: float = 0.0) -> None:
        """Report the result of a trial."""
        result = TrialResult(
            trial_id=trial_id,
            parameters=parameters,
            fitness=fitness,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
        )
        self._trials.append(result)

        if self._best_trial is None or fitness > self._best_trial.fitness:
            self._best_trial = result

    @property
    def best(self) -> Optional[TrialResult]:
        return self._best_trial

    @property
    def trial_count(self) -> int:
        return len(self._trials)

    def get_top_trials(self, n: int = 5) -> List[TrialResult]:
        return sorted(self._trials, key=lambda t: t.fitness, reverse=True)[:n]


class ForgeOptimizer:
    """
    FORGE uses Bayesian optimization to optimize APHELION strategy parameters.
    Lieutenant-tier (5 votes — influences parameters, not trade direction).

    Optimizable parameters:
    - Session trading windows
    - ARES vote threshold
    - SENTINEL risk parameters
    - Stop-loss/take-profit multipliers
    """

    DEFAULT_SPACE = [
        ParameterSpec("ares_threshold", 0.45, 0.70, 0.55, 0.01),
        ParameterSpec("base_risk_pct", 0.005, 0.025, 0.02, 0.001),
        ParameterSpec("kelly_fraction", 0.10, 0.50, 0.25, 0.05),
        ParameterSpec("sl_atr_multiplier", 1.0, 3.0, 1.5, 0.1),
        ParameterSpec("tp_atr_multiplier", 1.5, 5.0, 2.5, 0.1),
        ParameterSpec("asian_size_mult", 0.0, 1.0, 0.5, 0.1),
        ParameterSpec("london_size_mult", 0.5, 1.5, 1.2, 0.1),
        ParameterSpec("ny_overlap_size_mult", 0.5, 2.0, 1.5, 0.1),
    ]

    def __init__(self, parameter_space: Optional[List[ParameterSpec]] = None):
        self._optimizer = BayesianOptimizer(
            parameter_space or self.DEFAULT_SPACE
        )
        self._last_optimization: Optional[datetime] = None
        self._optimization_interval_trades = 2000

    def suggest_parameters(self) -> Dict[str, float]:
        return self._optimizer.suggest()

    def report_result(self, trial_id: int, params: Dict[str, float],
                      fitness: float, **kwargs) -> None:
        self._optimizer.report(trial_id, params, fitness, **kwargs)

    @property
    def best_parameters(self) -> Optional[Dict[str, float]]:
        if self._optimizer.best:
            return self._optimizer.best.parameters
        return None

    @property
    def trial_count(self) -> int:
        return self._optimizer.trial_count
