"""Tests for APHELION Kalman Filter Signal Smoother."""

import numpy as np
import pytest

from aphelion.filters.kalman import (
    KalmanConfig,
    KalmanFilter,
    KalmanState,
    AdaptiveKalmanSmoother,
)


# ─── KalmanConfig tests ──────────────────────────────────────────────────────


class TestKalmanConfig:

    def test_defaults(self):
        cfg = KalmanConfig()
        assert cfg.state_dim in (1, 2, 3)
        assert cfg.process_noise > 0
        assert cfg.measurement_noise > 0
        assert cfg.gate_sigma > 0

    def test_custom_config(self):
        cfg = KalmanConfig(state_dim=3, process_noise=0.5, measurement_noise=0.1)
        assert cfg.state_dim == 3
        assert cfg.process_noise == 0.5
        assert cfg.measurement_noise == 0.1


# ─── KalmanFilter tests ──────────────────────────────────────────────────────


class TestKalmanFilter:

    def test_single_update_returns_state(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1))
        state = kf.update(1.0)
        assert isinstance(state, KalmanState)
        assert isinstance(state.level, float)
        assert isinstance(state.uncertainty, float)

    def test_level_tracks_constant(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1, adapt_noise=False))
        for _ in range(50):
            state = kf.update(42.0)
        assert abs(state.level - 42.0) < 0.5

    def test_velocity_tracks_ramp(self):
        kf = KalmanFilter(KalmanConfig(state_dim=2, adapt_noise=False))
        for i in range(100):
            state = kf.update(float(i))
        assert abs(state.velocity - 1.0) < 0.2

    def test_acceleration_captures_quadratic(self):
        kf = KalmanFilter(KalmanConfig(state_dim=3, adapt_noise=False))
        for i in range(150):
            state = kf.update(float(i * i) * 0.01)
        assert abs(state.acceleration) > 1e-6

    def test_outlier_rejection(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1, gate_sigma=3.0, adapt_noise=False))
        for _ in range(50):
            kf.update(10.0)
        outlier_state = kf.update(1000.0)
        assert outlier_state.outlier is True
        assert outlier_state.level < 100.0

    def test_adaptive_noise(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1, adapt_noise=True, adaptation_window=20))
        rng = np.random.default_rng(42)
        for _ in range(100):
            state = kf.update(10.0 + float(rng.normal(0, 0.5)))
        assert state.uncertainty > 0.0

    def test_uncertainty_decreases_with_observations(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1, adapt_noise=False))
        u_first = kf.update(5.0).uncertainty
        for _ in range(20):
            kf.update(5.0)
        u_later = kf.update(5.0).uncertainty
        assert u_later < u_first

    def test_signal_to_noise_property(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1))
        state = kf.update(10.0)
        assert isinstance(state.signal_to_noise, float)
        assert state.signal_to_noise >= 0.0

    def test_trend_strength_property(self):
        kf = KalmanFilter(KalmanConfig(state_dim=2))
        for i in range(30):
            state = kf.update(float(i))
        assert state.trend_strength > 0.0

    def test_log_likelihood_updates(self):
        kf = KalmanFilter(KalmanConfig(state_dim=1))
        kf.update(5.0)
        state = kf.update(5.0)
        assert isinstance(state.log_likelihood, float)

    def test_state_vector_property(self):
        kf = KalmanFilter(KalmanConfig(state_dim=2))
        kf.update(10.0)
        assert kf.state_vector.shape == (2,)

    def test_covariance_property(self):
        kf = KalmanFilter(KalmanConfig(state_dim=3))
        kf.update(1.0)
        assert kf.covariance.shape == (3, 3)


# ─── AdaptiveKalmanSmoother tests ────────────────────────────────────────────


class TestAdaptiveKalmanSmoother:

    def test_smooth_single_channel(self):
        smoother = AdaptiveKalmanSmoother()
        results = []
        for i in range(50):
            val = 5.0 + np.sin(i * 0.1)
            state = smoother.smooth(val, channel="direction")
            results.append(state)
        assert len(results) == 50
        assert all(isinstance(s, KalmanState) for s in results)

    def test_smooth_signal_multi_channel(self):
        smoother = AdaptiveKalmanSmoother()
        result = smoother.smooth_signal(direction=0.5, confidence=0.8, uncertainty=0.2)
        assert isinstance(result, dict)
        assert "direction" in result
        assert "confidence" in result
        assert "uncertainty" in result
        assert all(isinstance(v, KalmanState) for v in result.values())

    def test_multiple_channels_independent(self):
        smoother = AdaptiveKalmanSmoother()
        for _ in range(20):
            smoother.smooth(10.0, channel="direction")
            smoother.smooth(0.5, channel="confidence")
        s1 = smoother.smooth(10.0, channel="direction")
        s2 = smoother.smooth(0.5, channel="confidence")
        assert abs(s1.level - 10.0) < 2.0
        assert abs(s2.level - 0.5) < 0.5

    def test_batch_smooth(self):
        smoother = AdaptiveKalmanSmoother()
        data = np.sin(np.linspace(0, 4 * np.pi, 100)) * 5
        states = smoother.batch_smooth(data)
        assert len(states) == 100
        assert all(isinstance(s, KalmanState) for s in states)

    def test_batch_rts_smooth(self):
        smoother = AdaptiveKalmanSmoother()
        data = np.sin(np.linspace(0, 4 * np.pi, 100)) * 5 + np.random.default_rng(42).normal(0, 0.5, 100)
        smoothed = smoother.batch_rts_smooth(data)
        assert len(smoothed) == 100
        raw_diff = np.diff(data)
        smooth_diff = np.diff(smoothed)
        assert np.std(smooth_diff) < np.std(raw_diff)

    def test_regime_change_detection(self):
        # gate_sigma=0 disables outlier gating so the large innovation is stored
        cfg = KalmanConfig(state_dim=1, adapt_noise=False, gate_sigma=0)
        smoother = AdaptiveKalmanSmoother(cfg)
        rng = np.random.default_rng(42)
        for _ in range(60):
            smoother.smooth(10.0 + float(rng.normal(0, 0.01)), channel="test")
        # Large regime shift
        smoother.smooth(500.0, channel="test")
        detected = smoother.detect_regime_change(channel="test")
        assert bool(detected) is True

    def test_reset_channel(self):
        smoother = AdaptiveKalmanSmoother()
        smoother.smooth(10.0, channel="a")
        smoother.reset(channel="a")
        assert "a" not in smoother._filters

    def test_reset_all(self):
        smoother = AdaptiveKalmanSmoother()
        smoother.smooth(10.0, channel="a")
        smoother.smooth(20.0, channel="b")
        smoother.reset()
        assert len(smoother._filters) == 0
