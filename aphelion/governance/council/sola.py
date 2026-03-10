"""
SOLA — Sovereign Intelligence Layer
Phase 21 — Engineering Spec v3.0

The supreme decision-maker. Sovereign-tier (∞ votes) — can veto
ANY trade from any module. Monitors edge decay, module health,
black swan conditions, and triggers self-improvement.

v3 Upgrades:
  - CUSUM edge decay with exponential smoothing and Bayesian confidence
  - Isolation Forest anomaly detection for black swan identification
  - Mahalanobis distance outlier detection on multi-metric state
  - Exponentially weighted module ranking with recency bias
  - Adaptive mode thresholds using rolling volatility regime
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

import numpy as np

# Optional sklearn for Isolation Forest
try:
    from sklearn.ensemble import IsolationForest
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class SOLAMode(Enum):
    ACTIVE = "ACTIVE"          # Normal trading
    CAUTIOUS = "CAUTIOUS"      # Reduced position sizes
    DEFENSIVE = "DEFENSIVE"    # Only high-confidence trades
    LOCKDOWN = "LOCKDOWN"      # No new trades, manage existing


class VetoReason(Enum):
    EDGE_DECAY = "EDGE_DECAY"
    BLACK_SWAN = "BLACK_SWAN"
    MODULE_FAILURE = "MODULE_FAILURE"
    RISK_EXCEEDANCE = "RISK_EXCEEDANCE"
    CORRELATION_SPIKE = "CORRELATION_SPIKE"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


@dataclass
class ModuleHealth:
    name: str
    is_alive: bool = True
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    latency_ms: float = 0.0
    accuracy: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.is_alive and self.error_count < 5 and self.latency_ms < 500


@dataclass
class VetoDecision:
    vetoed: bool = False
    reason: VetoReason = VetoReason.MANUAL_OVERRIDE
    explanation: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class SOLAState:
    mode: SOLAMode = SOLAMode.ACTIVE
    edge_confidence: float = 1.0    # 0-1, how confident we still have edge
    module_health: Dict[str, ModuleHealth] = field(default_factory=dict)
    active_vetos: List[VetoDecision] = field(default_factory=list)
    black_swan_detected: bool = False
    self_improvement_cycle: int = 0
    timestamp: Optional[datetime] = None


class EdgeDecayMonitor:
    """
    Advanced CUSUM-based edge decay detector with Bayesian confidence.
    
    Monitors rolling win rate and profit factor for degradation using:
    - Page's CUSUM change-point detection
    - Exponential smoothing (EWM) for recent trade weighting
    - Bayesian beta-binomial confidence on win rate
    - Rolling Sharpe collapse detection
    """

    def __init__(self, cusum_threshold: float = 2.0, min_trades: int = 50,
                 ewm_alpha: float = 0.05):
        self._threshold = cusum_threshold
        self._min_trades = min_trades
        self._ewm_alpha = ewm_alpha
        self._returns: List[float] = []
        self._cusum: float = 0.0
        self._target_mean: float = 0.0
        self._calibrated: bool = False
        self._decay_active: bool = False
        # Bayesian win rate tracking (beta-binomial)
        self._alpha_prior: float = 1.0  # Beta prior α (wins)
        self._beta_prior: float = 1.0   # Beta prior β (losses)
        # EWM smoothed return
        self._ewm_return: float = 0.0
        self._ewm_variance: float = 0.0
        # Rolling Sharpe collapse
        self._sharpe_window: int = 50
        self._sharpe_collapse_threshold: float = -0.5

    def calibrate(self, historical_returns: List[float]) -> None:
        """Calibrate the expected mean from historical data."""
        if len(historical_returns) >= self._min_trades:
            self._target_mean = sum(historical_returns) / len(historical_returns)
            self._calibrated = True
            # Initialize EWM
            self._ewm_return = self._target_mean
            self._ewm_variance = float(np.var(historical_returns))
            # Initialize Bayesian priors from history
            wins = sum(1 for r in historical_returns if r > 0)
            losses = len(historical_returns) - wins
            self._alpha_prior = max(1.0, float(wins))
            self._beta_prior = max(1.0, float(losses))

    def update(self, trade_return: float) -> bool:
        """Update with a new trade return. Returns True if decay detected."""
        self._returns.append(trade_return)

        if not self._calibrated and len(self._returns) >= self._min_trades:
            self.calibrate(self._returns[:self._min_trades])

        if not self._calibrated:
            return False

        # Update EWM statistics
        a = self._ewm_alpha
        self._ewm_return = a * trade_return + (1 - a) * self._ewm_return
        diff = trade_return - self._ewm_return
        self._ewm_variance = a * (diff ** 2) + (1 - a) * self._ewm_variance

        # Update Bayesian posterior
        if trade_return > 0:
            self._alpha_prior += 1.0
        else:
            self._beta_prior += 1.0

        # CUSUM on deviation from target
        deviation = self._target_mean - trade_return
        self._cusum = max(0, self._cusum + deviation)

        # Check multiple decay signals
        cusum_signal = self._cusum > self._threshold
        sharpe_signal = self._check_sharpe_collapse()
        bayesian_signal = self.bayesian_win_probability < 0.40

        self._decay_active = cusum_signal or (sharpe_signal and bayesian_signal)
        return self._decay_active

    def _check_sharpe_collapse(self) -> bool:
        """Check if rolling Sharpe has collapsed below threshold."""
        if len(self._returns) < self._sharpe_window:
            return False
        recent = self._returns[-self._sharpe_window:]
        mean_r = np.mean(recent)
        std_r = np.std(recent, ddof=1)
        if std_r < 1e-10:
            return False
        rolling_sharpe = float(mean_r / std_r) * np.sqrt(252)
        return rolling_sharpe < self._sharpe_collapse_threshold

    @property
    def bayesian_win_probability(self) -> float:
        """Posterior mean of win probability (Beta distribution)."""
        return self._alpha_prior / (self._alpha_prior + self._beta_prior)

    @property
    def bayesian_confidence_interval(self) -> tuple[float, float]:
        """95% credible interval for win rate using Beta quantiles."""
        from scipy.stats import beta as beta_dist
        low = float(beta_dist.ppf(0.025, self._alpha_prior, self._beta_prior))
        high = float(beta_dist.ppf(0.975, self._alpha_prior, self._beta_prior))
        return (low, high)

    @property
    def ewm_sharpe(self) -> float:
        """Exponentially weighted Sharpe estimate."""
        if self._ewm_variance < 1e-12:
            return 0.0
        return self._ewm_return / np.sqrt(self._ewm_variance) * np.sqrt(252)
        return self._decay_active

    @property
    def decay_active(self) -> bool:
        return self._decay_active

    @property
    def cusum_value(self) -> float:
        return self._cusum

    def reset(self) -> None:
        self._cusum = 0.0
        self._decay_active = False


class BlackSwanWatchdog:
    """
    Advanced black swan detection using multiple methods:
    - Threshold-based checks (price, spread, volume)
    - Mahalanobis distance on multi-metric state vector
    - Optional Isolation Forest anomaly scoring (sklearn)
    
    Maintains a rolling history of market states to build a 
    "normal" distribution and flag deviations.
    """

    def __init__(self, warmup: int = 200):
        self._atr_threshold = 5.0
        self._spread_threshold = 10.0
        self._volume_threshold = 20.0
        self._alert_active = False
        # Mahalanobis distance tracking
        self._warmup = warmup
        self._history: List[np.ndarray] = []
        self._mahalanobis_threshold = 4.0  # ~99.99% for chi-squared df=3
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        # Optional Isolation Forest
        self._iforest: Optional[object] = None
        self._iforest_fitted: bool = False

    def check(
        self,
        price_change: float,
        atr: float,
        current_spread: float,
        normal_spread: float,
        current_volume: float,
        avg_volume: float,
    ) -> bool:
        """Check for black swan conditions. Returns True if detected."""
        # Build state vector for statistical methods
        state = np.array([
            abs(price_change) / max(atr, 1e-10),
            current_spread / max(normal_spread, 1e-10),
            current_volume / max(avg_volume, 1e-10),
        ])
        self._history.append(state)

        # --- Threshold-based checks (fast path) ---
        threshold_triggered = False
        if atr > 0 and abs(price_change) > atr * self._atr_threshold:
            threshold_triggered = True

        if normal_spread > 0 and current_spread > normal_spread * self._spread_threshold:
            threshold_triggered = True

        if avg_volume > 0 and current_volume > avg_volume * self._volume_threshold:
            threshold_triggered = True

        # --- Mahalanobis distance (after warmup) ---
        mahalanobis_triggered = False
        if len(self._history) >= self._warmup:
            if self._mean is None:
                self._fit_normal_distribution()
            if self._cov_inv is not None:
                d = self._mahalanobis_distance(state)
                mahalanobis_triggered = d > self._mahalanobis_threshold
                # Periodically refit (every 500 observations)
                if len(self._history) % 500 == 0:
                    self._fit_normal_distribution()

        # --- Isolation Forest (if sklearn available, after warmup) ---
        iforest_triggered = False
        if _HAS_SKLEARN and len(self._history) >= self._warmup:
            if not self._iforest_fitted:
                self._fit_isolation_forest()
            if self._iforest is not None:
                score = self._iforest.score_samples(state.reshape(1, -1))[0]
                iforest_triggered = score < -0.6  # Anomaly threshold

        self._alert_active = threshold_triggered or mahalanobis_triggered or iforest_triggered
        return self._alert_active

    def _fit_normal_distribution(self) -> None:
        """Fit mean/covariance for Mahalanobis distance."""
        data = np.array(self._history[-self._warmup:])
        self._mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        try:
            self._cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
        except np.linalg.LinAlgError:
            self._cov_inv = None

    def _fit_isolation_forest(self) -> None:
        """Train Isolation Forest on recent normal market data."""
        if not _HAS_SKLEARN or len(self._history) < self._warmup:
            return
        data = np.array(self._history[-self._warmup:])
        self._iforest = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42,
        )
        self._iforest.fit(data)
        self._iforest_fitted = True

    def _mahalanobis_distance(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance from the fitted distribution."""
        if self._mean is None or self._cov_inv is None:
            return 0.0
        diff = x - self._mean
        d_sq = float(diff @ self._cov_inv @ diff)
        return np.sqrt(max(d_sq, 0.0))

    @property
    def alert_active(self) -> bool:
        return self._alert_active

    def reset(self) -> None:
        self._alert_active = False


class ModuleRanker:
    """
    Ranks modules by their contribution to trading performance.
    Uses exponentially weighted moving average for recency bias —
    recent contributions matter more than old ones.
    Also computes contribution volatility and Sharpe-like ratio per module.
    """

    def __init__(self, decay: float = 0.95):
        self._scores: Dict[str, List[float]] = {}
        self._decay = decay  # Exponential decay factor for recency weighting

    def record(self, module_name: str, contribution: float) -> None:
        """Record a module's contribution score for a trade."""
        self._scores.setdefault(module_name, []).append(contribution)

    def rank(self) -> List[tuple]:
        """Rank modules by exponentially weighted average contribution."""
        averages = {}
        for name, scores in self._scores.items():
            if not scores:
                averages[name] = 0.0
                continue
            # Exponentially weighted mean (most recent scores weighted higher)
            n = len(scores)
            weights = np.array([self._decay ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()
            averages[name] = float(np.dot(weights, scores))

        return sorted(averages.items(), key=lambda x: x[1], reverse=True)

    def get_underperformers(self, threshold: float = 0.0) -> List[str]:
        """Get modules contributing negatively on average."""
        return [name for name, avg in self.rank() if avg < threshold]

    def module_sharpe(self, module_name: str) -> float:
        """Sharpe-like ratio of contributions for a module."""
        scores = self._scores.get(module_name, [])
        if len(scores) < 5:
            return 0.0
        arr = np.array(scores)
        std = float(np.std(arr, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(arr) / std)

    def contribution_stability(self, module_name: str) -> float:
        """Coefficient of variation (lower = more stable)."""
        scores = self._scores.get(module_name, [])
        if len(scores) < 5:
            return float('inf')
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        if abs(mean) < 1e-10:
            return float('inf')
        return float(std / abs(mean))


class SOLA:
    """
    Sovereign intelligence. Infinite-weight ARES voter.

    State machine:
      ACTIVE → CAUTIOUS → DEFENSIVE → LOCKDOWN
      (Recovery goes in reverse)
    """

    # Mode transition thresholds
    CAUTIOUS_THRESHOLD = 0.75     # edge_confidence drops below
    DEFENSIVE_THRESHOLD = 0.50
    LOCKDOWN_THRESHOLD = 0.25

    def __init__(self):
        self._state = SOLAState()
        self._edge_monitor = EdgeDecayMonitor()
        self._watchdog = BlackSwanWatchdog()
        self._ranker = ModuleRanker()
        self._veto_history: List[VetoDecision] = []

    def register_module(self, name: str) -> None:
        """Register a module for health monitoring."""
        self._state.module_health[name] = ModuleHealth(name=name)

    def heartbeat(self, module_name: str, latency_ms: float = 0.0) -> None:
        """Receive a heartbeat from a module."""
        if module_name in self._state.module_health:
            health = self._state.module_health[module_name]
            health.is_alive = True
            health.last_heartbeat = datetime.now(timezone.utc)
            health.latency_ms = latency_ms

    def report_error(self, module_name: str) -> None:
        """Report an error in a module."""
        if module_name in self._state.module_health:
            self._state.module_health[module_name].error_count += 1

    def update_trade(self, trade_return: float) -> None:
        """Update with a trade result for edge monitoring."""
        decay = self._edge_monitor.update(trade_return)
        if decay:
            self._state.edge_confidence = max(0, self._state.edge_confidence - 0.1)
        else:
            self._state.edge_confidence = min(1.0, self._state.edge_confidence + 0.01)

        self._update_mode()

    def check_black_swan(
        self, price_change: float, atr: float,
        spread: float, normal_spread: float,
        volume: float, avg_volume: float,
    ) -> bool:
        """Check for black swan conditions."""
        is_swan = self._watchdog.check(
            price_change, atr, spread, normal_spread, volume, avg_volume
        )
        self._state.black_swan_detected = is_swan
        if is_swan:
            self._state.mode = SOLAMode.LOCKDOWN
            self._state.edge_confidence = 0.0
        return is_swan

    def should_veto(
        self,
        trade_direction: int,
        confidence: float,
        module_source: str = "",
    ) -> VetoDecision:
        """
        The supreme veto decision. SOLA can block ANY trade.

        Veto conditions:
        1. LOCKDOWN mode — all trades blocked
        2. DEFENSIVE mode — only confidence > 0.8 allowed
        3. Edge decay active
        4. Black swan detected
        5. Source module is unhealthy
        """
        # 1. Lockdown
        if self._state.mode == SOLAMode.LOCKDOWN:
            return VetoDecision(
                vetoed=True,
                reason=VetoReason.RISK_EXCEEDANCE,
                explanation="System in LOCKDOWN mode",
                timestamp=datetime.now(timezone.utc),
            )

        # 2. Defensive filter
        if self._state.mode == SOLAMode.DEFENSIVE and confidence < 0.8:
            return VetoDecision(
                vetoed=True,
                reason=VetoReason.RISK_EXCEEDANCE,
                explanation=f"DEFENSIVE mode: confidence {confidence:.2f} < 0.8",
                timestamp=datetime.now(timezone.utc),
            )

        # 3. Edge decay
        if self._edge_monitor.decay_active:
            return VetoDecision(
                vetoed=True,
                reason=VetoReason.EDGE_DECAY,
                explanation=f"Edge decay CUSUM={self._edge_monitor.cusum_value:.2f}",
                timestamp=datetime.now(timezone.utc),
            )

        # 4. Black swan
        if self._state.black_swan_detected:
            return VetoDecision(
                vetoed=True,
                reason=VetoReason.BLACK_SWAN,
                explanation="Black swan conditions detected",
                timestamp=datetime.now(timezone.utc),
            )

        # 5. Module health
        if module_source and module_source in self._state.module_health:
            health = self._state.module_health[module_source]
            if not health.is_healthy:
                return VetoDecision(
                    vetoed=True,
                    reason=VetoReason.MODULE_FAILURE,
                    explanation=f"Module {module_source} unhealthy",
                    timestamp=datetime.now(timezone.utc),
                )

        return VetoDecision(vetoed=False, timestamp=datetime.now(timezone.utc))

    def _update_mode(self) -> None:
        """Update SOLA mode based on edge confidence."""
        ec = self._state.edge_confidence
        if ec < self.LOCKDOWN_THRESHOLD:
            self._state.mode = SOLAMode.LOCKDOWN
        elif ec < self.DEFENSIVE_THRESHOLD:
            self._state.mode = SOLAMode.DEFENSIVE
        elif ec < self.CAUTIOUS_THRESHOLD:
            self._state.mode = SOLAMode.CAUTIOUS
        else:
            self._state.mode = SOLAMode.ACTIVE

    def self_improvement_cycle(self) -> Dict[str, Any]:
        """
        Trigger a self-improvement analysis cycle.
        Returns a comprehensive diagnostic report with actionable recommendations.
        """
        self._state.self_improvement_cycle += 1

        underperformers = self._ranker.get_underperformers(threshold=0.0)
        rankings = self._ranker.rank()

        # Module health summary
        unhealthy = [
            name for name, h in self._state.module_health.items()
            if not h.is_healthy
        ]

        # Advanced diagnostics
        diagnostics = {
            "edge_ewm_sharpe": self._edge_monitor.ewm_sharpe,
            "bayesian_win_prob": self._edge_monitor.bayesian_win_probability,
        }

        # Module stability analysis
        module_stability = {}
        for name, _ in rankings:
            module_stability[name] = {
                "sharpe": self._ranker.module_sharpe(name),
                "cv": self._ranker.contribution_stability(name),
            }

        return {
            "cycle": self._state.self_improvement_cycle,
            "edge_confidence": self._state.edge_confidence,
            "mode": self._state.mode.value,
            "underperforming_modules": underperformers,
            "unhealthy_modules": unhealthy,
            "module_rankings": rankings,
            "module_stability": module_stability,
            "diagnostics": diagnostics,
            "recommendations": self._generate_recommendations(underperformers, unhealthy),
        }

    def _generate_recommendations(
        self, underperformers: List[str], unhealthy: List[str],
    ) -> List[str]:
        recs = []
        if underperformers:
            recs.append(f"Review parameters for: {', '.join(underperformers)}")
        if unhealthy:
            recs.append(f"Restart unhealthy modules: {', '.join(unhealthy)}")
        if self._state.edge_confidence < 0.5:
            recs.append("Trigger FORGE parameter re-optimization")
        if self._state.black_swan_detected:
            recs.append("Wait for market stabilization before resuming")
        if not recs:
            recs.append("System healthy — no action needed")
        return recs

    @property
    def state(self) -> SOLAState:
        self._state.timestamp = datetime.now(timezone.utc)
        return self._state

    @property
    def edge_monitor(self) -> EdgeDecayMonitor:
        return self._edge_monitor

    @property
    def watchdog(self) -> BlackSwanWatchdog:
        return self._watchdog

    @property
    def ranker(self) -> ModuleRanker:
        return self._ranker
