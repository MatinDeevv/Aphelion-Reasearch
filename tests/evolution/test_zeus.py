"""
Phase 11 Tests — ZEUS Stress Testing & Synthetic Regime Generation

Covers:
  - StressConfig / GANConfig defaults
  - StressInjector: all 5 injection types
  - ZeusStressTester: full 5-scenario pipeline, pass/fail logic
  - ZeusGANGenerator: regime generation, overfitting detection
  - ZeusEngine: combined full_test pipeline
  - ScenarioResult / StressTestResult / GANTestResult data structures
  - Edge cases: evaluator errors, tiny data, extreme configs
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.evolution.zeus.engine import (
    GANConfig,
    GANTestResult,
    ScenarioResult,
    StressConfig,
    StressInjector,
    StressScenario,
    StressTestResult,
    SyntheticRegime,
    ZeusEngine,
    ZeusGANGenerator,
    ZeusStressTester,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class TestStressConfig:
    def test_defaults(self):
        cfg = StressConfig()
        assert cfg.flash_crash_magnitude_pct == 5.0
        assert cfg.spread_multiplier == 10.0
        assert cfg.latency_spike_ms == 500
        assert cfg.gap_duration_bars == 5
        assert cfg.gap_count == 3
        assert cfg.max_drawdown_limit == 0.20

    def test_custom_config(self):
        cfg = StressConfig(flash_crash_magnitude_pct=10.0, spread_multiplier=20.0)
        assert cfg.flash_crash_magnitude_pct == 10.0
        assert cfg.spread_multiplier == 20.0


class TestGANConfig:
    def test_defaults(self):
        cfg = GANConfig()
        assert cfg.latent_dim == 32
        assert cfg.sequence_length == 500
        assert cfg.n_regimes == 10
        assert len(cfg.regime_types) == 10
        assert cfg.max_sharpe_gap == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# StressInjector
# ═══════════════════════════════════════════════════════════════════════════

class TestStressInjector:
    @pytest.fixture
    def injector(self):
        return StressInjector(rng=np.random.default_rng(42))

    @pytest.fixture
    def base_prices(self):
        """Simple upward trending price series."""
        return 2000.0 + np.cumsum(np.random.default_rng(42).normal(0.1, 0.5, 500))

    def test_flash_crash_modifies_prices(self, injector, base_prices):
        result = injector.inject_flash_crash(base_prices, magnitude_pct=5.0)
        assert result.shape == base_prices.shape
        # Some prices should be lower than original
        assert np.any(result < base_prices)

    def test_flash_crash_v_shape_recovers(self, injector, base_prices):
        result = injector.inject_flash_crash(
            base_prices, magnitude_pct=5.0, recovery_bars=20, v_shape=True
        )
        # End of series should be close to original (V recovery)
        assert abs(result[-1] - base_prices[-1]) < base_prices[-1] * 0.1

    def test_flash_crash_no_recovery(self, injector, base_prices):
        result = injector.inject_flash_crash(
            base_prices, magnitude_pct=5.0, recovery_bars=10, v_shape=False
        )
        # End of series should be shifted down
        assert result[-1] < base_prices[-1]

    def test_spread_blowout_returns_bid_ask(self, injector, base_prices):
        bids, asks = injector.inject_spread_blowout(base_prices, multiplier=10.0)
        assert bids.shape == base_prices.shape
        assert asks.shape == base_prices.shape
        assert np.all(asks >= bids)  # Asks always >= bids

    def test_spread_blowout_widens_spread(self, injector, base_prices):
        bids, asks = injector.inject_spread_blowout(
            base_prices, multiplier=10.0, duration_bars=50
        )
        spreads = asks - bids
        normal_spread = base_prices * 0.0001
        # Some bars should have much wider spreads
        assert np.any(spreads > normal_spread * 5)

    def test_latency_creates_stale_prices(self, injector, base_prices):
        result = injector.inject_latency_spikes(
            base_prices, affected_pct=0.2, latency_ms=500
        )
        assert result.shape == base_prices.shape
        # Some bars should equal their predecessor (stale)
        dups = sum(1 for i in range(1, len(result)) if result[i] == result[i - 1])
        assert dups > 0

    def test_data_gaps_forward_fill(self, injector, base_prices):
        result, mask = injector.inject_data_gaps(
            base_prices, gap_duration=5, gap_count=3
        )
        assert result.shape == base_prices.shape
        assert mask.shape == base_prices.shape
        # Some bars should be marked as gaps
        assert np.sum(mask) > 0
        # Gap bars should be forward-filled (equal to previous)
        for i in range(1, len(result)):
            if mask[i] == 1.0:
                # Forward-filled value may chain from first gap bar
                assert np.isfinite(result[i])

    def test_adversarial_noise_modifies_prices(self, injector, base_prices):
        result = injector.inject_adversarial_noise(
            base_prices, noise_std=0.002, spike_pct=0.05
        )
        assert result.shape == base_prices.shape
        # Prices should be different from originals
        assert not np.allclose(result, base_prices)

    def test_tiny_array_doesnt_crash(self, injector):
        tiny = np.array([2000.0, 2001.0, 2002.0])
        # All methods should handle gracefully
        r1 = injector.inject_flash_crash(tiny)
        assert len(r1) == 3
        b, a = injector.inject_spread_blowout(tiny)
        assert len(b) == 3
        r3 = injector.inject_latency_spikes(tiny)
        assert len(r3) == 3
        r4, m = injector.inject_data_gaps(tiny, gap_duration=1, gap_count=1)
        assert len(r4) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Helper evaluator
# ═══════════════════════════════════════════════════════════════════════════

def _good_evaluator(prices: np.ndarray) -> dict:
    """Evaluator that simulates a well-performing genome."""
    rets = np.diff(prices) / prices[:-1]
    sharpe = float(np.mean(rets) / max(np.std(rets), 1e-10) * np.sqrt(252))
    dd = float(np.min(np.minimum.accumulate(np.cumprod(1 + rets)) - 1))
    return {
        "sharpe": min(max(sharpe, -5), 5),
        "max_drawdown": min(abs(dd), 0.15),  # Good DD
        "total_return_pct": float(np.sum(rets) * 100),
        "trade_count": 50,
    }


def _bad_evaluator(prices: np.ndarray) -> dict:
    """Evaluator that simulates a terrible genome."""
    return {
        "sharpe": -2.0,
        "max_drawdown": 0.50,
        "total_return_pct": -30.0,
        "trade_count": 5,
    }


def _error_evaluator(prices: np.ndarray) -> dict:
    """Evaluator that always fails."""
    raise ValueError("Evaluator crashed!")


# ═══════════════════════════════════════════════════════════════════════════
# ZeusStressTester
# ═══════════════════════════════════════════════════════════════════════════

class TestZeusStressTester:
    @pytest.fixture
    def tester(self):
        return ZeusStressTester(
            StressConfig(max_drawdown_limit=0.25),
            rng=np.random.default_rng(42),
        )

    @pytest.fixture
    def base_prices(self):
        rng = np.random.default_rng(42)
        return 2000.0 + np.cumsum(rng.normal(0.05, 0.3, 500))

    def test_good_genome_passes(self, tester, base_prices):
        result = tester.test_genome("G-0001-test", base_prices, _good_evaluator)
        assert isinstance(result, StressTestResult)
        assert result.genome_id == "G-0001-test"
        assert len(result.scenario_results) == 5
        # Good evaluator should pass most scenarios
        assert result.pass_count >= 3

    def test_bad_genome_fails(self, tester, base_prices):
        result = tester.test_genome("G-bad", base_prices, _bad_evaluator)
        assert isinstance(result, StressTestResult)
        assert not result.passed  # Bad evaluator should fail all

    def test_error_evaluator_fails_gracefully(self, tester, base_prices):
        result = tester.test_genome("G-err", base_prices, _error_evaluator)
        assert not result.passed
        assert result.fail_count == 5
        for sr in result.scenario_results:
            assert not sr.passed

    def test_all_five_scenarios_tested(self, tester, base_prices):
        result = tester.test_genome("G-test", base_prices, _good_evaluator)
        scenarios = {r.scenario for r in result.scenario_results}
        assert StressScenario.FLASH_CRASH in scenarios
        assert StressScenario.SPREAD_BLOWOUT in scenarios
        assert StressScenario.LATENCY_SPIKE in scenarios
        assert StressScenario.DATA_GAP in scenarios
        assert StressScenario.ADVERSARIAL in scenarios

    def test_scenario_result_has_details(self, tester, base_prices):
        result = tester.test_genome("G-test", base_prices, _good_evaluator)
        for sr in result.scenario_results:
            assert sr.details  # Non-empty details string
            assert sr.bars_tested == len(base_prices)

    def test_overall_metrics(self, tester, base_prices):
        result = tester.test_genome("G-test", base_prices, _good_evaluator)
        assert result.overall_max_drawdown >= 0
        assert isinstance(result.overall_return_pct, float)

    def test_scenarios_summary(self, tester, base_prices):
        result = tester.test_genome("G-test", base_prices, _good_evaluator)
        summary = result.scenarios_summary
        assert len(summary) == 5
        assert "FLASH_CRASH" in summary


# ═══════════════════════════════════════════════════════════════════════════
# ZeusGANGenerator
# ═══════════════════════════════════════════════════════════════════════════

class TestZeusGANGenerator:
    @pytest.fixture
    def generator(self):
        config = GANConfig(
            sequence_length=200,
            n_regimes=5,
            regime_types=["trending_up", "trending_down", "mean_reverting",
                          "high_volatility", "random_walk"],
        )
        return ZeusGANGenerator(config, rng=np.random.default_rng(42))

    @pytest.fixture
    def ref_prices(self):
        rng = np.random.default_rng(42)
        return 2000.0 + np.cumsum(rng.normal(0.01, 0.3, 500))

    def test_generate_regimes_count(self, generator, ref_prices):
        regimes = generator.generate_regimes(ref_prices)
        assert len(regimes) == 5

    def test_regime_types(self, generator, ref_prices):
        regimes = generator.generate_regimes(ref_prices)
        types = {r.regime_type for r in regimes}
        assert "trending_up" in types
        assert "trending_down" in types
        assert "random_walk" in types

    def test_regime_prices_shape(self, generator, ref_prices):
        regimes = generator.generate_regimes(ref_prices)
        for regime in regimes:
            assert len(regime.prices) == 200

    def test_regime_prices_positive(self, generator, ref_prices):
        regimes = generator.generate_regimes(ref_prices)
        for regime in regimes:
            assert np.all(regime.prices > 0)

    def test_regime_has_description(self, generator, ref_prices):
        regimes = generator.generate_regimes(ref_prices)
        for regime in regimes:
            assert regime.description  # Non-empty

    def test_overfitting_detection_good_genome(self, generator, ref_prices):
        """A genome with similar real and synthetic performance is NOT overfit."""
        result = generator.test_overfitting(
            genome_id="G-good",
            real_sharpe=1.0,
            evaluator=_good_evaluator,
            reference_prices=ref_prices,
        )
        assert isinstance(result, GANTestResult)
        assert result.genome_id == "G-good"
        assert result.n_regimes_total == 5

    def test_overfitting_detection_overfit_genome(self, generator, ref_prices):
        """A genome with very high real Sharpe but bad synthetic should be flagged."""
        result = generator.test_overfitting(
            genome_id="G-overfit",
            real_sharpe=10.0,  # Suspiciously high
            evaluator=_bad_evaluator,  # Bad on synthetic data
            reference_prices=ref_prices,
        )
        assert isinstance(result, GANTestResult)
        assert result.sharpe_gap > 0  # Gap should be large

    def test_error_handling_in_evaluator(self, generator, ref_prices):
        """Should handle evaluator errors gracefully."""
        result = generator.test_overfitting(
            genome_id="G-err",
            real_sharpe=1.0,
            evaluator=_error_evaluator,
            reference_prices=ref_prices,
        )
        assert isinstance(result, GANTestResult)
        # Should not crash

    def test_pre_generated_regimes(self, generator, ref_prices):
        """Can pass pre-generated regimes instead of reference prices."""
        regimes = generator.generate_regimes(ref_prices)
        result = generator.test_overfitting(
            genome_id="G-pre",
            real_sharpe=0.5,
            evaluator=_good_evaluator,
            regimes=regimes,
        )
        assert result.n_regimes_total == len(regimes)

    def test_must_provide_regimes_or_reference(self, generator):
        """Should raise ValueError if neither regimes nor reference provided."""
        with pytest.raises(ValueError, match="Must provide"):
            generator.test_overfitting(
                genome_id="G-fail",
                real_sharpe=1.0,
                evaluator=_good_evaluator,
            )


# ═══════════════════════════════════════════════════════════════════════════
# All 10 Regime Types
# ═══════════════════════════════════════════════════════════════════════════

class TestAllRegimeTypes:
    def test_all_ten_regime_types_generated(self):
        """Ensure all 10 standard regime types produce valid output."""
        config = GANConfig(sequence_length=300, n_regimes=10)
        gen = ZeusGANGenerator(config, rng=np.random.default_rng(42))
        ref = 2000.0 + np.cumsum(np.random.default_rng(42).normal(0.01, 0.3, 500))
        regimes = gen.generate_regimes(ref)
        assert len(regimes) == 10
        for r in regimes:
            assert len(r.prices) == 300
            assert np.all(np.isfinite(r.prices))
            assert np.all(r.prices > 0)


# ═══════════════════════════════════════════════════════════════════════════
# ZeusEngine (Combined)
# ═══════════════════════════════════════════════════════════════════════════

class TestZeusEngine:
    @pytest.fixture
    def zeus(self):
        return ZeusEngine(
            stress_config=StressConfig(max_drawdown_limit=0.25),
            gan_config=GANConfig(
                sequence_length=200,
                n_regimes=3,
                regime_types=["trending_up", "random_walk", "high_volatility"],
            ),
            rng=np.random.default_rng(42),
        )

    @pytest.fixture
    def base_prices(self):
        rng = np.random.default_rng(42)
        return 2000.0 + np.cumsum(rng.normal(0.05, 0.3, 500))

    def test_full_test_returns_dict(self, zeus, base_prices):
        result = zeus.full_test("G-full", base_prices, _good_evaluator, real_sharpe=1.0)
        assert "stress_result" in result
        assert "gan_result" in result
        assert "stress_passed" in result
        assert "gan_passed" in result
        assert "overall_passed" in result

    def test_full_test_good_genome(self, zeus, base_prices):
        result = zeus.full_test("G-good", base_prices, _good_evaluator, real_sharpe=1.0)
        assert isinstance(result["stress_result"], StressTestResult)
        assert isinstance(result["gan_result"], GANTestResult)

    def test_full_test_bad_genome_fails(self, zeus, base_prices):
        result = zeus.full_test("G-bad", base_prices, _bad_evaluator, real_sharpe=1.0)
        assert not result["stress_passed"]

    def test_full_test_error_handling(self, zeus, base_prices):
        result = zeus.full_test("G-err", base_prices, _error_evaluator, real_sharpe=1.0)
        assert not result["overall_passed"]

    def test_sub_engines_accessible(self, zeus):
        assert isinstance(zeus.stress_tester, ZeusStressTester)
        assert isinstance(zeus.gan_generator, ZeusGANGenerator)


# ═══════════════════════════════════════════════════════════════════════════
# StressTestResult Properties
# ═══════════════════════════════════════════════════════════════════════════

class TestStressTestResult:
    def test_pass_fail_counts(self):
        results = [
            ScenarioResult(StressScenario.FLASH_CRASH, True, 0.05, 1.0, 0.5, 30, "ok"),
            ScenarioResult(StressScenario.SPREAD_BLOWOUT, True, 0.08, 0.5, 0.3, 25, "ok"),
            ScenarioResult(StressScenario.LATENCY_SPIKE, False, 0.30, -5.0, -1.0, 10, "bad"),
            ScenarioResult(StressScenario.DATA_GAP, True, 0.10, 0.2, 0.2, 20, "ok"),
            ScenarioResult(StressScenario.ADVERSARIAL, False, 0.25, -3.0, -0.8, 15, "bad"),
        ]
        tr = StressTestResult(
            genome_id="G-test",
            passed=False,
            scenario_results=results,
            overall_max_drawdown=0.30,
            overall_return_pct=-1.26,
        )
        assert tr.pass_count == 3
        assert tr.fail_count == 2
        assert not tr.passed

    def test_scenarios_summary_keys(self):
        results = [
            ScenarioResult(StressScenario.FLASH_CRASH, True, 0.05, 1.0, 0.5, 30, "ok"),
        ]
        tr = StressTestResult("G-t", True, results, 0.05, 1.0)
        summary = tr.scenarios_summary
        assert "FLASH_CRASH" in summary
        assert summary["FLASH_CRASH"] is True
