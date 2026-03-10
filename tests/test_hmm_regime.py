"""Tests for APHELION HMM Regime Detection."""

import numpy as np
import pytest

from aphelion.macro.hmm_regime import (
    HMMConfig,
    HMMRegimeDetector,
    HMMRegimeLabel,
    HMMRegimeState,
)


# ─── HMMConfig tests ─────────────────────────────────────────────────────────


class TestHMMConfig:

    def test_defaults(self):
        cfg = HMMConfig()
        assert cfg.n_states == 4
        assert cfg.n_iter == 100
        assert cfg.min_observations == 200

    def test_custom(self):
        cfg = HMMConfig(n_states=3, min_observations=50)
        assert cfg.n_states == 3
        assert cfg.min_observations == 50


# ─── HMMRegimeLabel tests ────────────────────────────────────────────────────


class TestHMMRegimeLabel:

    def test_all_labels_exist(self):
        labels = [l.value for l in HMMRegimeLabel]
        assert "BULL_QUIET" in labels
        assert "BEAR_VOLATILE" in labels
        assert "CRISIS" in labels
        assert "UNKNOWN" in labels


# ─── HMMRegimeDetector tests ─────────────────────────────────────────────────


class TestHMMRegimeDetector:

    @staticmethod
    def _make_synthetic_data(n: int = 500, seed: int = 42):
        """Generate returns + volatilities mimicking regime switches."""
        rng = np.random.default_rng(seed)
        returns = np.empty(n)
        vols = np.empty(n)
        volumes = np.empty(n)
        for i in range(n):
            if i < n // 3:
                returns[i] = rng.normal(0.001, 0.005)
                vols[i] = abs(rng.normal(0.005, 0.001))
                volumes[i] = rng.lognormal(10, 0.3)
            elif i < 2 * n // 3:
                returns[i] = rng.normal(-0.002, 0.02)
                vols[i] = abs(rng.normal(0.02, 0.005))
                volumes[i] = rng.lognormal(11, 0.5)
            else:
                returns[i] = rng.normal(0.0015, 0.01)
                vols[i] = abs(rng.normal(0.01, 0.003))
                volumes[i] = rng.lognormal(10.5, 0.4)
        return returns, vols, volumes

    def test_fit_basic(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        success = detector.fit(returns, vols, volumes)
        assert success is True
        assert detector._is_fitted is True

    def test_fit_without_volume(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100, use_volume=False))
        returns, vols, _ = self._make_synthetic_data()
        success = detector.fit(returns, vols)
        assert success is True

    def test_update_returns_state(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        state = detector.update(0.001, 0.01, volume=50000.0)
        assert isinstance(state, HMMRegimeState)
        assert isinstance(state.regime_label, HMMRegimeLabel)
        assert 0 <= state.confidence <= 1.0

    def test_decode_sequence(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        path = detector.decode_sequence(returns, vols, volumes)
        assert len(path) == len(returns)
        assert all(isinstance(int(s), int) for s in path)

    def test_regime_probabilities_sum_to_one(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        state = detector.update(0.0, 0.01, volume=50000.0)
        prob_sum = sum(state.regime_probabilities)
        assert abs(prob_sum - 1.0) < 1e-4

    def test_regime_duration_positive(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        for _ in range(10):
            state = detector.update(0.001, 0.01, volume=50000.0)
        assert state.regime_duration >= 1

    def test_auto_fit_on_buffer_full(self):
        cfg = HMMConfig(n_states=2, min_observations=50)
        detector = HMMRegimeDetector(cfg)
        rng = np.random.default_rng(99)
        for i in range(60):
            r = float(rng.normal(0, 0.01))
            v = abs(float(rng.normal(0.01, 0.003)))
            state = detector.update(r, v, volume=50000.0)
        assert detector._is_fitted is True
        assert isinstance(state, HMMRegimeState)

    def test_regime_info(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        info = detector.regime_info()
        assert isinstance(info, list)
        assert len(info) == 3
        assert "label" in info[0]
        assert "mean_return" in info[0]

    def test_update_before_fit_returns_unfitted_state(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=1000))
        state = detector.update(0.001, 0.01)
        assert state.is_fitted is False
        assert state.regime_label == HMMRegimeLabel.UNKNOWN

    def test_multiple_detectors_independent(self):
        d1 = HMMRegimeDetector(HMMConfig(n_states=2, min_observations=100))
        d2 = HMMRegimeDetector(HMMConfig(n_states=4, min_observations=100))
        returns, vols, volumes = self._make_synthetic_data()
        d1.fit(returns, vols, volumes)
        d2.fit(returns, vols, volumes)
        s1 = d1.update(0.001, 0.01, volume=50000.0)
        s2 = d2.update(0.001, 0.01, volume=50000.0)
        assert isinstance(s1, HMMRegimeState)
        assert isinstance(s2, HMMRegimeState)

    def test_is_fitted_property(self):
        detector = HMMRegimeDetector()
        assert detector.is_fitted is False
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        assert detector.is_fitted is True

    def test_transition_matrix_property(self):
        detector = HMMRegimeDetector(HMMConfig(n_states=3, min_observations=100))
        assert detector.transition_matrix is None
        returns, vols, volumes = self._make_synthetic_data()
        detector.fit(returns, vols, volumes)
        tm = detector.transition_matrix
        assert tm is not None
        assert tm.shape == (3, 3)
        # Rows should sum to ~1
        for row in tm:
            assert abs(row.sum() - 1.0) < 1e-4
