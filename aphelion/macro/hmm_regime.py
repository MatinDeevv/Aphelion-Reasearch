"""
APHELION Hidden Markov Model Regime Detection

Learns hidden market states from the data itself — no hardcoded ADX/ATR thresholds.
Finds regimes you didn't know existed by modelling the joint distribution of
returns, volatility, and volume as emissions from latent states.

Model:
  - GaussianHMM with N hidden states (default 4)
  - Observation features: [log-return, realised vol, volume z-score, spread]
  - Transition matrix reveals regime persistence and switching probabilities
  - Viterbi path gives MAP regime sequence; forward-backward gives marginals
  - Online regime tracking via filtered state probabilities (no full refit)

Interpretability:
  - After fitting, each state is auto-labelled by its emission means:
    high-vol + negative-return → "CRISIS",  low-vol + positive → "BULL_QUIET", etc.
  - Transition matrix gives expected regime duration: 1 / (1 - a_ii) bars

Dependencies:
  - hmmlearn (optional): pip install hmmlearn
  - Falls back to a simple numpy EM implementation if hmmlearn unavailable

References:
  - Rabiner (1989) "A Tutorial on Hidden Markov Models"
  - Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary
    Time Series and the Business Cycle"
  - Bulla & Bulla (2006) "Stylized Facts of Financial Time Series and HMM"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False

logger = logging.getLogger(__name__)


# ─── Data types ───────────────────────────────────────────────────────────────


class HMMRegimeLabel(str, Enum):
    """Auto-assigned regime labels based on emission characteristics."""
    BULL_QUIET = "BULL_QUIET"         # Positive drift, low vol
    BULL_VOLATILE = "BULL_VOLATILE"   # Positive drift, high vol
    BEAR_QUIET = "BEAR_QUIET"         # Negative drift, low vol
    BEAR_VOLATILE = "BEAR_VOLATILE"   # Negative drift, high vol
    CRISIS = "CRISIS"                 # Extreme vol, negative drift
    UNKNOWN = "UNKNOWN"


@dataclass
class HMMRegimeState:
    """Output of the HMM regime detector for a single timestep."""
    current_regime: int = 0                        # Raw state index
    regime_label: HMMRegimeLabel = HMMRegimeLabel.UNKNOWN
    regime_probabilities: list[float] = field(default_factory=list)  # P(state_i | obs)
    regime_duration: float = 0.0                   # Expected bars in current regime
    transition_probability: float = 0.0            # P(switching out next bar)
    confidence: float = 0.0                        # Max probability among states
    is_fitted: bool = False


@dataclass
class HMMConfig:
    """Configuration for the HMM regime detector."""
    n_states: int = 4                  # Number of hidden states
    n_iter: int = 100                  # EM iterations for fitting
    covariance_type: str = "full"      # "full", "diag", "tied", "spherical"
    min_observations: int = 200        # Minimum bars before fitting
    refit_interval: int = 500          # Refit every N new observations
    random_seed: int = 42
    # Observation features to use
    use_volume: bool = True
    use_spread: bool = False


# ─── HMM Regime Detector ─────────────────────────────────────────────────────


class HMMRegimeDetector:
    """
    Gaussian HMM regime detector that learns market states from data.

    Usage::

        detector = HMMRegimeDetector(HMMConfig(n_states=4))

        # Feed historical data to fit
        detector.fit(returns, volatilities, volumes)

        # Online: update with each new bar
        state = detector.update(current_return, current_vol, current_volume)
        print(state.regime_label, state.confidence)

        # Inspect learned regimes
        for info in detector.regime_info():
            print(info)
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self._cfg = config or HMMConfig()
        self._model: Optional[object] = None
        self._is_fitted: bool = False
        self._labels: dict[int, HMMRegimeLabel] = {}

        # Observation buffer for online accumulation
        self._obs_buffer: list[np.ndarray] = []
        self._last_state: int = 0
        self._state_durations: dict[int, float] = {}
        self._obs_since_fit: int = 0

    def fit(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        spreads: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Fit the HMM on historical observation features.

        Args:
            returns: Log returns (or simple returns), shape (T,)
            volatilities: Realised volatility estimates, shape (T,)
            volumes: Volume z-scores (optional), shape (T,)
            spreads: Bid-ask spread (optional), shape (T,)

        Returns:
            True if fitting succeeded.
        """
        X = self._build_features(returns, volatilities, volumes, spreads)
        if len(X) < self._cfg.min_observations:
            logger.warning(
                "HMM fit requires %d observations, got %d",
                self._cfg.min_observations, len(X),
            )
            return False

        try:
            if _HAS_HMMLEARN:
                self._model = GaussianHMM(
                    n_components=self._cfg.n_states,
                    covariance_type=self._cfg.covariance_type,
                    n_iter=self._cfg.n_iter,
                    random_state=self._cfg.random_seed,
                )
                self._model.fit(X)
            else:
                self._model = _SimpleGaussianHMM(
                    n_states=self._cfg.n_states,
                    n_iter=self._cfg.n_iter,
                    seed=self._cfg.random_seed,
                )
                self._model.fit(X)

            self._is_fitted = True
            self._auto_label_states()
            self._compute_durations()
            self._obs_since_fit = 0
            logger.info("HMM fitted with %d states on %d observations", self._cfg.n_states, len(X))
            return True

        except Exception as e:
            logger.error("HMM fitting failed: %s", e)
            return False

    def update(
        self,
        ret: float,
        volatility: float,
        volume: Optional[float] = None,
        spread: Optional[float] = None,
    ) -> HMMRegimeState:
        """
        Online update: classify a single new observation.

        Returns HMMRegimeState with current regime, probabilities, and confidence.
        """
        obs = self._build_single_obs(ret, volatility, volume, spread)
        self._obs_buffer.append(obs)
        self._obs_since_fit += 1

        if not self._is_fitted:
            # Try auto-fit when we have enough data
            if len(self._obs_buffer) >= self._cfg.min_observations:
                X = np.array(self._obs_buffer)
                self.fit(
                    X[:, 0], X[:, 1],
                    X[:, 2] if X.shape[1] > 2 else None,
                    X[:, 3] if X.shape[1] > 3 else None,
                )
            return HMMRegimeState()

        # Periodic refit
        if self._obs_since_fit >= self._cfg.refit_interval:
            X = np.array(self._obs_buffer[-self._cfg.min_observations * 3:])
            self.fit(
                X[:, 0], X[:, 1],
                X[:, 2] if X.shape[1] > 2 else None,
                X[:, 3] if X.shape[1] > 3 else None,
            )

        # Predict current state probabilities
        try:
            obs_2d = obs.reshape(1, -1)
            if _HAS_HMMLEARN:
                probs = self._model.predict_proba(obs_2d)[0]
                state = int(np.argmax(probs))
            else:
                probs = self._model.predict_proba(obs_2d)[0]
                state = int(np.argmax(probs))

            self._last_state = state
            label = self._labels.get(state, HMMRegimeLabel.UNKNOWN)
            duration = self._state_durations.get(state, 0.0)

            # Transition probability = 1 - self-transition
            trans_prob = 1.0 - self._get_self_transition(state)

            return HMMRegimeState(
                current_regime=state,
                regime_label=label,
                regime_probabilities=probs.tolist(),
                regime_duration=duration,
                transition_probability=trans_prob,
                confidence=float(np.max(probs)),
                is_fitted=True,
            )
        except Exception as e:
            logger.warning("HMM predict failed: %s", e)
            return HMMRegimeState()

    def decode_sequence(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Viterbi decode: returns MAP state sequence for the full observation array.
        Useful for backtest regime labelling.
        """
        if not self._is_fitted:
            return np.zeros(len(returns), dtype=int)

        X = self._build_features(returns, volatilities, volumes)
        if _HAS_HMMLEARN:
            return self._model.predict(X)
        else:
            return self._model.predict(X)

    def regime_info(self) -> list[dict]:
        """
        Return interpretable information about each learned regime:
        mean return, mean volatility, expected duration, label.
        """
        if not self._is_fitted:
            return []

        info = []
        if _HAS_HMMLEARN:
            means = self._model.means_
        else:
            means = self._model.means_

        for i in range(self._cfg.n_states):
            info.append({
                "state": i,
                "label": self._labels.get(i, HMMRegimeLabel.UNKNOWN).value,
                "mean_return": float(means[i, 0]),
                "mean_volatility": float(means[i, 1]),
                "expected_duration_bars": self._state_durations.get(i, 0.0),
                "self_transition_prob": self._get_self_transition(i),
            })
        return info

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _build_features(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        spreads: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stack observation features into (T, n_features) array."""
        cols = [returns.reshape(-1, 1), volatilities.reshape(-1, 1)]
        if self._cfg.use_volume and volumes is not None:
            cols.append(volumes.reshape(-1, 1))
        if self._cfg.use_spread and spreads is not None:
            cols.append(spreads.reshape(-1, 1))
        return np.hstack(cols)

    def _build_single_obs(
        self,
        ret: float,
        volatility: float,
        volume: Optional[float] = None,
        spread: Optional[float] = None,
    ) -> np.ndarray:
        """Build a single observation vector."""
        obs = [ret, volatility]
        if self._cfg.use_volume and volume is not None:
            obs.append(volume)
        if self._cfg.use_spread and spread is not None:
            obs.append(spread)
        return np.array(obs)

    def _auto_label_states(self) -> None:
        """
        Assign interpretable labels to each hidden state based on emission means.
        Uses the mean return (col 0) and mean volatility (col 1).
        """
        if not self._is_fitted:
            return

        if _HAS_HMMLEARN:
            means = self._model.means_
        else:
            means = self._model.means_

        # Sort states by volatility
        vol_median = np.median(means[:, 1])
        ret_median = np.median(means[:, 0])

        for i in range(self._cfg.n_states):
            m_ret = means[i, 0]
            m_vol = means[i, 1]

            high_vol = m_vol > vol_median
            positive_ret = m_ret > ret_median

            # Check for extreme vol (crisis)
            vol_spread = np.std(means[:, 1])
            if m_vol > vol_median + 1.5 * vol_spread and m_ret < ret_median:
                self._labels[i] = HMMRegimeLabel.CRISIS
            elif positive_ret and not high_vol:
                self._labels[i] = HMMRegimeLabel.BULL_QUIET
            elif positive_ret and high_vol:
                self._labels[i] = HMMRegimeLabel.BULL_VOLATILE
            elif not positive_ret and not high_vol:
                self._labels[i] = HMMRegimeLabel.BEAR_QUIET
            elif not positive_ret and high_vol:
                self._labels[i] = HMMRegimeLabel.BEAR_VOLATILE
            else:
                self._labels[i] = HMMRegimeLabel.UNKNOWN

    def _compute_durations(self) -> None:
        """Compute expected regime duration from transition matrix diagonal."""
        if not self._is_fitted:
            return
        if _HAS_HMMLEARN:
            transmat = self._model.transmat_
        else:
            transmat = self._model.transmat_

        for i in range(self._cfg.n_states):
            p_stay = transmat[i, i]
            if p_stay < 1.0:
                self._state_durations[i] = 1.0 / (1.0 - p_stay)
            else:
                self._state_durations[i] = float("inf")

    def _get_self_transition(self, state: int) -> float:
        if not self._is_fitted:
            return 0.0
        if _HAS_HMMLEARN:
            return float(self._model.transmat_[state, state])
        else:
            return float(self._model.transmat_[state, state])

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        if not self._is_fitted:
            return None
        if _HAS_HMMLEARN:
            return self._model.transmat_.copy()
        return self._model.transmat_.copy()


# ─── Fallback HMM implementation ─────────────────────────────────────────────


class _SimpleGaussianHMM:
    """
    Minimal Gaussian HMM with EM fitting when hmmlearn is not installed.
    Supports fit, predict, predict_proba, and Viterbi decoding.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 100, seed: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self._rng = np.random.default_rng(seed)
        self.means_: Optional[np.ndarray] = None      # (K, D)
        self.covars_: Optional[np.ndarray] = None      # (K, D, D)
        self.transmat_: Optional[np.ndarray] = None    # (K, K)
        self.startprob_: Optional[np.ndarray] = None   # (K,)

    def fit(self, X: np.ndarray) -> None:
        """Baum-Welch EM algorithm for Gaussian emissions."""
        T, D = X.shape
        K = self.n_states

        # Initialise with K-means-style partitioning
        self.startprob_ = np.ones(K) / K
        self.transmat_ = np.ones((K, K)) / K + np.eye(K) * 0.5
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # K-means init for means
        indices = self._rng.choice(T, size=K, replace=False)
        self.means_ = X[indices].copy()
        self.covars_ = np.array([np.eye(D) * np.var(X, axis=0) for _ in range(K)])

        for iteration in range(self.n_iter):
            # E-step: forward-backward
            log_obs = self._log_obs_prob(X)  # (T, K)
            log_alpha = self._forward(log_obs)
            log_beta = self._backward(log_obs)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)  # (T, K)

            # Xi: transition posteriors
            log_xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        log_xi[t, i, j] = (
                            log_alpha[t, i]
                            + np.log(self.transmat_[i, j] + 1e-300)
                            + log_obs[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                log_xi[t] -= _logsumexp(log_xi[t].ravel())

            xi = np.exp(log_xi)

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()

            for i in range(K):
                denom = xi[:, i, :].sum()
                if denom > 1e-10:
                    self.transmat_[i] = xi[:, i, :].sum(axis=0) / denom
                else:
                    self.transmat_[i] = np.ones(K) / K

            for k in range(K):
                nk = gamma[:, k].sum()
                if nk > 1e-10:
                    self.means_[k] = (gamma[:, k, None] * X).sum(axis=0) / nk
                    diff = X - self.means_[k]
                    self.covars_[k] = (gamma[:, k, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / nk
                    # Regularise
                    self.covars_[k] += np.eye(D) * 1e-4

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding."""
        T, D = X.shape
        K = self.n_states
        log_obs = self._log_obs_prob(X)

        viterbi = np.zeros((T, K))
        backptr = np.zeros((T, K), dtype=int)
        viterbi[0] = np.log(self.startprob_ + 1e-300) + log_obs[0]

        for t in range(1, T):
            for j in range(K):
                scores = viterbi[t - 1] + np.log(self.transmat_[:, j] + 1e-300)
                backptr[t, j] = int(np.argmax(scores))
                viterbi[t, j] = scores[backptr[t, j]] + log_obs[t, j]

        # Backtrace
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(viterbi[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]
        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Forward-backward → filtered state probabilities."""
        log_obs = self._log_obs_prob(X)
        log_alpha = self._forward(log_obs)
        log_beta = self._backward(log_obs)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def _log_obs_prob(self, X: np.ndarray) -> np.ndarray:
        """Log emission probabilities: p(x_t | state_k)."""
        T, D = X.shape
        K = self.n_states
        log_p = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means_[k]
            cov = self.covars_[k]
            try:
                sign, logdet = np.linalg.slogdet(cov)
                cov_inv = np.linalg.inv(cov)
                mahal = np.sum(diff @ cov_inv * diff, axis=1)
                log_p[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
            except np.linalg.LinAlgError:
                log_p[:, k] = -1e10
        return log_p

    def _forward(self, log_obs: np.ndarray) -> np.ndarray:
        T, K = log_obs.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_obs[0]
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + np.log(self.transmat_[:, j] + 1e-300))
                    + log_obs[t, j]
                )
        return log_alpha

    def _backward(self, log_obs: np.ndarray) -> np.ndarray:
        T, K = log_obs.shape
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for j in range(K):
                log_beta[t, j] = _logsumexp(
                    np.log(self.transmat_[j] + 1e-300) + log_obs[t + 1] + log_beta[t + 1]
                )
        return log_beta


def _logsumexp(x: np.ndarray, axis: Optional[int] = None,
               keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        if axis is not None:
            result = np.squeeze(result, axis=axis)
        else:
            result = result.ravel()
            if result.size == 1:
                result = result[0]
    return result
