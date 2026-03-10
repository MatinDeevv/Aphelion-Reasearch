"""
APHELION Kalman Filter Signal Smoother

Dynamic state estimation that adapts its noise model in real-time.
Far superior to EMA for filtering HYDRA's raw signal because it:
  1. Maintains an explicit uncertainty estimate (not just a point value)
  2. Adapts process/measurement noise online via innovation monitoring
  3. Provides velocity (trend) and acceleration estimates alongside level
  4. Detects regime changes when innovations exceed expected bounds

Models:
  - KalmanFilter: Core 1D Kalman with configurable state dimension
  - AdaptiveKalmanSmoother: Wraps KalmanFilter with online noise adaptation
    using Mehra's innovation-based estimator and outlier gating.

State vector: [level, velocity, acceleration] (configurable 1-3 dims)
Observation: scalar price/signal

References:
  - Kalman (1960) "A New Approach to Linear Filtering"
  - Mehra (1972) "Approaches to Adaptive Filtering" (innovation-based Q/R adapt)
  - Harvey (1989) "Forecasting, Structural Time Series Models"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class KalmanConfig:
    """Configuration for the Kalman filter."""

    # State dimension: 1=level only, 2=level+velocity, 3=level+velocity+accel
    state_dim: int = 2

    # Initial process noise (how fast the underlying signal can change)
    process_noise: float = 1e-4

    # Initial measurement noise (how noisy the observations are)
    measurement_noise: float = 1e-2

    # Adaptive noise estimation
    adapt_noise: bool = True
    adaptation_window: int = 50        # Innovation window for noise estimation
    adaptation_rate: float = 0.05      # Exponential smoothing for Q/R updates

    # Outlier gating: reject observations > gate_sigma standard deviations
    gate_sigma: float = 4.0

    # Minimum noise floors to prevent filter divergence
    min_process_noise: float = 1e-8
    min_measurement_noise: float = 1e-6


@dataclass
class KalmanState:
    """Observable state output from the Kalman filter."""

    level: float = 0.0             # Filtered estimate of the signal
    velocity: float = 0.0          # Rate of change (trend)
    acceleration: float = 0.0      # Rate of change of velocity
    uncertainty: float = 1.0       # Estimation uncertainty (sqrt of P[0,0])
    innovation: float = 0.0        # Last prediction error
    innovation_var: float = 1.0    # Variance of innovation
    log_likelihood: float = 0.0    # Incremental log-likelihood
    outlier: bool = False          # Was last observation gated as outlier?
    process_noise: float = 1e-4    # Current adapted Q diagonal
    measurement_noise: float = 1e-2  # Current adapted R

    @property
    def signal_to_noise(self) -> float:
        """Ratio of |level| to uncertainty — higher = more confident."""
        if self.uncertainty < 1e-12:
            return float("inf")
        return abs(self.level) / self.uncertainty

    @property
    def trend_strength(self) -> float:
        """Normalised velocity: velocity / uncertainty."""
        if self.uncertainty < 1e-12:
            return 0.0
        return self.velocity / self.uncertainty


# ─── Core Kalman Filter ──────────────────────────────────────────────────────


class KalmanFilter:
    """
    Linear Kalman Filter with configurable state dimension.

    State transition model (constant-velocity or constant-acceleration):
        x_{t+1} = F @ x_t + w,   w ~ N(0, Q)
    Observation model:
        z_t = H @ x_t + v,       v ~ N(0, R)

    Usage::

        kf = KalmanFilter(KalmanConfig(state_dim=2))
        for price in prices:
            state = kf.update(price)
            print(state.level, state.velocity)
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self._cfg = config or KalmanConfig()
        n = self._cfg.state_dim
        assert 1 <= n <= 3, "state_dim must be 1, 2, or 3"

        # State vector and covariance
        self._x = np.zeros(n)       # [level, (velocity), (acceleration)]
        self._P = np.eye(n) * 1.0   # Initial uncertainty

        # Transition matrix (discrete constant-velocity / accel model, dt=1)
        self._F = np.eye(n)
        if n >= 2:
            self._F[0, 1] = 1.0     # level += velocity
        if n >= 3:
            self._F[1, 2] = 1.0     # velocity += acceleration
            self._F[0, 2] = 0.5     # level += 0.5 * acceleration

        # Observation matrix: we observe only the level
        self._H = np.zeros((1, n))
        self._H[0, 0] = 1.0

        # Noise matrices
        self._Q = np.eye(n) * self._cfg.process_noise
        self._R = np.array([[self._cfg.measurement_noise]])

        # Innovation tracking for adaptive noise
        self._innovations: list[float] = []
        self._init = False
        self._step = 0

    def predict(self) -> None:
        """Time-update (predict) step."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def _innovation(self, z: float) -> tuple[float, float]:
        """Compute innovation and its variance."""
        y = z - float((self._H @ self._x).item())               # innovation
        S = float((self._H @ self._P @ self._H.T).item() + self._R[0, 0])  # innovation variance
        return y, max(S, 1e-12)

    def _is_outlier(self, y: float, S: float) -> bool:
        """Mahalanobis gating for outlier detection."""
        if self._cfg.gate_sigma <= 0:
            return False
        return abs(y) > self._cfg.gate_sigma * np.sqrt(S)

    def correct(self, z: float) -> KalmanState:
        """Measurement-update (correct) step."""
        y, S = self._innovation(z)
        outlier = self._is_outlier(y, S)

        if outlier:
            # Skip update — keep predicted state but log the outlier
            return self._build_state(y, S, outlier=True)

        # Kalman gain
        K = (self._P @ self._H.T) / S  # (n, 1) / scalar
        self._x = self._x + K.ravel() * y
        I_KH = np.eye(len(self._x)) - K @ self._H
        # Joseph form for numerical stability
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        # Track innovation for adaptive noise
        self._innovations.append(y)
        if len(self._innovations) > self._cfg.adaptation_window:
            self._innovations = self._innovations[-self._cfg.adaptation_window:]

        return self._build_state(y, S, outlier=False)

    def update(self, z: float) -> KalmanState:
        """Full predict→correct cycle. Primary API."""
        if not self._init:
            self._x[0] = z
            self._init = True
            return self._build_state(0.0, self._R[0, 0], outlier=False)

        self._step += 1
        self.predict()
        state = self.correct(z)

        # Adaptive noise estimation (Mehra's innovation-based method)
        if self._cfg.adapt_noise and len(self._innovations) >= 10:
            self._adapt_noise()

        return state

    def _adapt_noise(self) -> None:
        """
        Mehra (1972) innovation-based noise adaptation.
        R ← sample variance of innovations
        Q ← back-solve from innovation covariance
        """
        inn = np.array(self._innovations[-self._cfg.adaptation_window:])
        alpha = self._cfg.adaptation_rate

        # Estimated measurement noise from innovation variance
        inn_var = float(np.var(inn))
        predicted_var = float((self._H @ self._P @ self._H.T).item())
        new_R = inn_var  # In steady state, S ≈ R + H P H^T, so R ≈ inn_var - HPH^T
        new_R = max(new_R, self._cfg.min_measurement_noise)

        # Exponential smoothing
        self._R[0, 0] = (1 - alpha) * self._R[0, 0] + alpha * new_R

        # Process noise: increase Q if innovations are systematically too large
        expected_S = predicted_var + self._R[0, 0]
        innovation_ratio = inn_var / max(expected_S, 1e-12)
        if innovation_ratio > 1.5:
            # Innovations bigger than expected → model is too smooth, increase Q
            scale = min(innovation_ratio, 3.0)
            self._Q *= (1 - alpha) + alpha * scale
        elif innovation_ratio < 0.5:
            # Innovations smaller than expected → model is too noisy, decrease Q
            scale = max(innovation_ratio, 0.3)
            self._Q *= (1 - alpha) + alpha * scale

        # Floor
        np.clip(self._Q, self._cfg.min_process_noise, None, out=self._Q)

    def _build_state(self, innovation: float, inn_var: float,
                     outlier: bool) -> KalmanState:
        n = len(self._x)
        ll = -0.5 * (np.log(2 * np.pi * inn_var) + innovation ** 2 / inn_var)
        return KalmanState(
            level=float(self._x[0]),
            velocity=float(self._x[1]) if n >= 2 else 0.0,
            acceleration=float(self._x[2]) if n >= 3 else 0.0,
            uncertainty=float(np.sqrt(self._P[0, 0])),
            innovation=innovation,
            innovation_var=inn_var,
            log_likelihood=float(ll),
            outlier=outlier,
            process_noise=float(self._Q[0, 0]),
            measurement_noise=float(self._R[0, 0]),
        )

    @property
    def state_vector(self) -> np.ndarray:
        """Raw state vector [level, velocity, acceleration]."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """State covariance matrix."""
        return self._P.copy()


# ─── Adaptive Kalman Smoother (production wrapper) ───────────────────────────


class AdaptiveKalmanSmoother:
    """
    Production-ready signal smoother for HYDRA output.

    Wraps KalmanFilter and provides:
      - Smooth signal with confidence bands
      - Regime change detection (large innovations)
      - Multi-signal support (can run independent filters for price, confidence, etc.)
      - Batch processing for backtest (filter + RTS smoother)

    Usage::

        smoother = AdaptiveKalmanSmoother()
        for signal in hydra_signals:
            result = smoother.smooth(signal.confidence)
            filtered = result.level
            trend = result.velocity
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self._cfg = config or KalmanConfig()
        self._filters: dict[str, KalmanFilter] = {}
        self._regime_change_threshold: float = 3.0  # std devs

    def _get_filter(self, channel: str = "default") -> KalmanFilter:
        if channel not in self._filters:
            self._filters[channel] = KalmanFilter(self._cfg)
        return self._filters[channel]

    def smooth(self, value: float, channel: str = "default") -> KalmanState:
        """Filter a single observation. Returns the filtered state."""
        kf = self._get_filter(channel)
        return kf.update(value)

    def smooth_signal(
        self,
        direction: float,
        confidence: float,
        uncertainty: float,
    ) -> dict[str, KalmanState]:
        """
        Smooth a full HYDRA signal (direction, confidence, uncertainty).
        Returns dict of KalmanState per channel.
        """
        return {
            "direction": self.smooth(direction, channel="direction"),
            "confidence": self.smooth(confidence, channel="confidence"),
            "uncertainty": self.smooth(uncertainty, channel="uncertainty"),
        }

    def detect_regime_change(self, channel: str = "default") -> bool:
        """
        Returns True if recent innovation suggests a regime change
        (signal has shifted suddenly beyond what the model expected).
        """
        kf = self._get_filter(channel)
        if not kf._innovations:
            return False
        last_inn = kf._innovations[-1]
        inn_std = np.std(kf._innovations) if len(kf._innovations) > 5 else 1.0
        if inn_std < 1e-12:
            return False
        return abs(last_inn) > self._regime_change_threshold * inn_std

    def batch_smooth(self, observations: np.ndarray) -> list[KalmanState]:
        """
        Forward pass on an array of observations.
        Returns list of KalmanState for each timestep.
        """
        kf = KalmanFilter(self._cfg)
        states: list[KalmanState] = []
        for z in observations:
            states.append(kf.update(float(z)))
        return states

    def batch_rts_smooth(self, observations: np.ndarray) -> np.ndarray:
        """
        Rauch-Tung-Striebel (RTS) fixed-interval smoother.
        Runs forward Kalman pass, then backward smoothing pass.
        Returns array of smoothed levels — optimal for backtest feature generation.
        """
        cfg = self._cfg
        kf = KalmanFilter(cfg)
        n = len(observations)
        dim = cfg.state_dim

        # Forward pass — store predicted and filtered states
        x_pred = np.zeros((n, dim))
        P_pred = np.zeros((n, dim, dim))
        x_filt = np.zeros((n, dim))
        P_filt = np.zeros((n, dim, dim))

        for t in range(n):
            if t == 0:
                kf._x[0] = observations[0]
                kf._init = True
            else:
                kf.predict()
                x_pred[t] = kf._x.copy()
                P_pred[t] = kf._P.copy()
                kf.correct(float(observations[t]))

            x_filt[t] = kf._x.copy()
            P_filt[t] = kf._P.copy()

        # Backward (RTS) smoothing pass
        x_smooth = np.zeros((n, dim))
        x_smooth[-1] = x_filt[-1]

        for t in range(n - 2, -1, -1):
            P_pred_t = P_pred[t + 1]
            det = np.linalg.det(P_pred_t)
            if abs(det) < 1e-20:
                x_smooth[t] = x_filt[t]
                continue
            C = P_filt[t] @ kf._F.T @ np.linalg.inv(P_pred_t)
            x_smooth[t] = x_filt[t] + C @ (x_smooth[t + 1] - x_pred[t + 1])

        return x_smooth[:, 0]  # Return smoothed levels

    def reset(self, channel: Optional[str] = None) -> None:
        """Reset one or all channels."""
        if channel:
            self._filters.pop(channel, None)
        else:
            self._filters.clear()
