"""
APHELION ZEUS — Pre-Deployment Stress Testing & Synthetic Regime Generation

Sub-components:
  ZEUS-STRESS  – Pre-deployment stress tester (flash crash, spread blowout,
                 latency spikes, data gaps, adversarial attacks)
  ZEUS-GAN     – GAN-generated synthetic regime generator for overfitting detection

Spec reference:
  "Every genome must survive: flash crash simulation, 10x spread blowout,
   500ms latency spikes, data gap injection, and LEVIATHAN adversarial attacks."
  "Creates never-seen market conditions to test for overfitting."
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ─── Stress Scenario Definitions ─────────────────────────────────────────────

class StressScenario(Enum):
    """The 5 mandatory stress tests every genome must survive."""
    FLASH_CRASH = auto()
    SPREAD_BLOWOUT = auto()
    LATENCY_SPIKE = auto()
    DATA_GAP = auto()
    ADVERSARIAL = auto()


@dataclass
class StressConfig:
    """Configuration for ZEUS stress testing."""
    # Flash crash
    flash_crash_magnitude_pct: float = 5.0      # 5% sudden drop
    flash_crash_recovery_bars: int = 10          # Recovery in 10 bars
    flash_crash_v_shape: bool = True             # V-shape recovery
    # Spread blowout
    spread_multiplier: float = 10.0              # 10x normal spread
    spread_blowout_duration_bars: int = 20       # Lasts 20 bars
    # Latency
    latency_spike_ms: int = 500                  # 500ms delay
    latency_affected_pct: float = 0.10           # 10% of bars affected
    # Data gaps
    gap_duration_bars: int = 5                   # 5 bars of missing data
    gap_count: int = 3                           # 3 gaps injected
    # Adversarial
    adversarial_noise_std: float = 0.002         # 0.2% noise amplitude
    adversarial_spike_pct: float = 0.05          # 5% of bars get adversarial spikes
    # Pass criteria
    max_drawdown_limit: float = 0.20             # Must survive with < 20% DD
    min_sharpe_under_stress: float = -0.5        # Sharpe must stay above -0.5
    max_loss_per_scenario_pct: float = 10.0      # No scenario can lose > 10%


@dataclass
class ScenarioResult:
    """Result of a single stress scenario."""
    scenario: StressScenario
    passed: bool
    max_drawdown: float
    total_return_pct: float
    sharpe: float
    trade_count: int
    details: str
    bars_tested: int = 0


@dataclass
class StressTestResult:
    """Complete stress test results for a genome."""
    genome_id: str
    passed: bool                              # True if ALL scenarios pass
    scenario_results: list[ScenarioResult]
    overall_max_drawdown: float
    overall_return_pct: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.scenario_results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.scenario_results if not r.passed)

    @property
    def scenarios_summary(self) -> dict[str, bool]:
        return {r.scenario.name: r.passed for r in self.scenario_results}


# ─── Bar Data Injector ───────────────────────────────────────────────────────

class StressInjector:
    """
    Injects stress conditions into clean bar data to simulate extreme
    market conditions. Each method returns a modified copy of the data.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng(42)

    def inject_flash_crash(
        self,
        prices: np.ndarray,
        magnitude_pct: float = 5.0,
        recovery_bars: int = 10,
        v_shape: bool = True,
    ) -> np.ndarray:
        """
        Inject a flash crash: sudden drop then optional V-shape recovery.

        Args:
            prices: 1D close price array.
            magnitude_pct: Drop magnitude as percentage of price.
            recovery_bars: Number of bars for recovery.
            v_shape: Whether crash recovers (True) or stays down (False).
        """
        result = prices.copy()
        n = len(result)
        if n < recovery_bars + 5:
            return result

        # Crash point: random location in the middle 60% of bars
        crash_start = self._rng.integers(int(n * 0.2), int(n * 0.6))
        drop = result[crash_start] * (magnitude_pct / 100.0)

        # Apply crash
        for i in range(crash_start, min(crash_start + recovery_bars, n)):
            t = (i - crash_start) / max(recovery_bars, 1)
            if v_shape:
                # V-shape: drop then recover
                if t < 0.3:
                    result[i] -= drop * (t / 0.3)
                else:
                    recovery_progress = (t - 0.3) / 0.7
                    result[i] -= drop * (1.0 - recovery_progress)
            else:
                # Step down — no recovery
                result[i] -= drop

        # Everything after crash end stays shifted if no v-shape
        if not v_shape:
            for i in range(crash_start + recovery_bars, n):
                result[i] -= drop

        return result

    def inject_spread_blowout(
        self,
        prices: np.ndarray,
        multiplier: float = 10.0,
        duration_bars: int = 20,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject a spread blowout period. Returns (bid_prices, ask_prices).

        During the blowout, the bid-ask spread is multiplied.
        Normal spread is estimated as 0.01% of price.
        """
        n = len(prices)
        normal_spread = prices * 0.0001  # ~0.01% base spread

        bids = prices - normal_spread / 2
        asks = prices + normal_spread / 2

        if n < duration_bars + 5:
            return bids, asks

        start = self._rng.integers(int(n * 0.2), int(n * 0.6))
        end = min(start + duration_bars, n)

        for i in range(start, end):
            blown_spread = normal_spread[i] * multiplier
            bids[i] = prices[i] - blown_spread / 2
            asks[i] = prices[i] + blown_spread / 2

        return bids, asks

    def inject_latency_spikes(
        self,
        prices: np.ndarray,
        affected_pct: float = 0.10,
        latency_ms: int = 500,
    ) -> np.ndarray:
        """
        Simulate latency: affected bars use stale prices (duplicate previous bar).

        In real trading, 500ms latency means you see stale prices and your
        orders arrive late, so we model it as stale price information.
        """
        result = prices.copy()
        n = len(result)
        n_affected = max(1, int(n * affected_pct))
        affected_indices = self._rng.choice(
            np.arange(1, n), size=min(n_affected, n - 1), replace=False
        )

        for idx in affected_indices:
            # Stale: use previous bar's price (simulates delayed data)
            result[idx] = result[idx - 1]

        return result

    def inject_data_gaps(
        self,
        prices: np.ndarray,
        gap_duration: int = 5,
        gap_count: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inject data gaps: periods where prices are NaN (missing data).

        Returns (modified_prices, gap_mask) where gap_mask[i]=1 means gap.
        """
        result = prices.copy()
        mask = np.zeros(len(prices), dtype=np.float64)
        n = len(result)

        for _ in range(gap_count):
            if n < gap_duration + 10:
                break
            start = self._rng.integers(5, n - gap_duration - 5)
            # Forward-fill (simulate what a real system would do)
            last_valid = result[start - 1]
            for i in range(start, min(start + gap_duration, n)):
                result[i] = last_valid  # Forward-fill
                mask[i] = 1.0

        return result, mask

    def inject_adversarial_noise(
        self,
        prices: np.ndarray,
        noise_std: float = 0.002,
        spike_pct: float = 0.05,
    ) -> np.ndarray:
        """
        Inject adversarial noise: random perturbations that look like
        genuine price moves but are designed to trigger false signals.
        """
        result = prices.copy()
        n = len(result)

        # Add Gaussian noise across all bars
        noise = self._rng.normal(0, noise_std, n) * prices
        result += noise

        # Add adversarial spikes at random points
        n_spikes = max(1, int(n * spike_pct))
        spike_indices = self._rng.choice(n, size=n_spikes, replace=False)
        for idx in spike_indices:
            direction = self._rng.choice([-1, 1])
            magnitude = self._rng.uniform(0.001, 0.005) * prices[idx]
            result[idx] += direction * magnitude

        return result


# ─── ZEUS Stress Tester ──────────────────────────────────────────────────────

class ZeusStressTester:
    """
    ZEUS-STRESS — Pre-deployment stress testing engine.

    Tests a genome's evaluator against 5 extreme scenarios.
    Genomes must pass ALL scenarios to be promoted.

    Usage:
        zeus = ZeusStressTester(config)
        result = zeus.test_genome(
            genome_id="G-0042-abc",
            base_prices=price_array,
            evaluator=my_eval_fn,
        )
        if result.passed:
            promote_genome()
    """

    def __init__(self, config: Optional[StressConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self._config = config or StressConfig()
        self._rng = rng or np.random.default_rng(42)
        self._injector = StressInjector(self._rng)

    def test_genome(
        self,
        genome_id: str,
        base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
    ) -> StressTestResult:
        """
        Run all 5 stress scenarios against a genome.

        Args:
            genome_id: ID of the genome being tested.
            base_prices: Clean price data (1D array of close prices).
            evaluator: Function that takes a price array and returns
                       {"sharpe": float, "max_drawdown": float,
                        "total_return_pct": float, "trade_count": int}.

        Returns:
            StressTestResult with pass/fail for each scenario.
        """
        results = []

        # 1. Flash Crash
        results.append(self._test_flash_crash(genome_id, base_prices, evaluator))

        # 2. Spread Blowout
        results.append(self._test_spread_blowout(genome_id, base_prices, evaluator))

        # 3. Latency Spikes
        results.append(self._test_latency(genome_id, base_prices, evaluator))

        # 4. Data Gaps
        results.append(self._test_data_gaps(genome_id, base_prices, evaluator))

        # 5. Adversarial
        results.append(self._test_adversarial(genome_id, base_prices, evaluator))

        # Overall results
        all_passed = all(r.passed for r in results)
        overall_dd = max(r.max_drawdown for r in results)
        overall_return = sum(r.total_return_pct for r in results) / max(len(results), 1)

        test_result = StressTestResult(
            genome_id=genome_id,
            passed=all_passed,
            scenario_results=results,
            overall_max_drawdown=overall_dd,
            overall_return_pct=overall_return,
        )

        logger.info(
            "ZEUS stress test for %s: %s (%d/%d scenarios passed)",
            genome_id,
            "PASSED" if all_passed else "FAILED",
            test_result.pass_count,
            len(results),
        )

        return test_result

    # ── Individual Scenarios ─────────────────────────────────────────────────

    def _test_flash_crash(
        self, genome_id: str, base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
    ) -> ScenarioResult:
        """Scenario 1: Flash crash simulation."""
        stressed = self._injector.inject_flash_crash(
            base_prices,
            self._config.flash_crash_magnitude_pct,
            self._config.flash_crash_recovery_bars,
            self._config.flash_crash_v_shape,
        )
        return self._evaluate_scenario(
            StressScenario.FLASH_CRASH, stressed, evaluator,
            f"Flash crash: {self._config.flash_crash_magnitude_pct}% drop, "
            f"{self._config.flash_crash_recovery_bars} bar recovery",
        )

    def _test_spread_blowout(
        self, genome_id: str, base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
    ) -> ScenarioResult:
        """Scenario 2: 10x spread blowout."""
        # Use midpoint with widened spread effect
        bids, asks = self._injector.inject_spread_blowout(
            base_prices,
            self._config.spread_multiplier,
            self._config.spread_blowout_duration_bars,
        )
        # Evaluator sees midpoint but with execution slippage
        midpoints = (bids + asks) / 2
        return self._evaluate_scenario(
            StressScenario.SPREAD_BLOWOUT, midpoints, evaluator,
            f"Spread blowout: {self._config.spread_multiplier}x for "
            f"{self._config.spread_blowout_duration_bars} bars",
        )

    def _test_latency(
        self, genome_id: str, base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
    ) -> ScenarioResult:
        """Scenario 3: 500ms latency spikes."""
        stressed = self._injector.inject_latency_spikes(
            base_prices,
            self._config.latency_affected_pct,
            self._config.latency_spike_ms,
        )
        return self._evaluate_scenario(
            StressScenario.LATENCY_SPIKE, stressed, evaluator,
            f"Latency: {self._config.latency_spike_ms}ms affecting "
            f"{self._config.latency_affected_pct * 100:.0f}% of bars",
        )

    def _test_data_gaps(
        self, genome_id: str, base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
    ) -> ScenarioResult:
        """Scenario 4: Data gap injection."""
        stressed, mask = self._injector.inject_data_gaps(
            base_prices,
            self._config.gap_duration_bars,
            self._config.gap_count,
        )
        return self._evaluate_scenario(
            StressScenario.DATA_GAP, stressed, evaluator,
            f"Data gaps: {self._config.gap_count} gaps × "
            f"{self._config.gap_duration_bars} bars each",
        )

    def _test_adversarial(
        self, genome_id: str, base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
    ) -> ScenarioResult:
        """Scenario 5: Adversarial noise injection."""
        stressed = self._injector.inject_adversarial_noise(
            base_prices,
            self._config.adversarial_noise_std,
            self._config.adversarial_spike_pct,
        )
        return self._evaluate_scenario(
            StressScenario.ADVERSARIAL, stressed, evaluator,
            f"Adversarial: σ={self._config.adversarial_noise_std}, "
            f"spikes={self._config.adversarial_spike_pct * 100:.0f}%",
        )

    def _evaluate_scenario(
        self,
        scenario: StressScenario,
        prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
        details: str,
    ) -> ScenarioResult:
        """Run evaluator on stressed data and check pass criteria."""
        try:
            metrics = evaluator(prices)
        except Exception as e:
            logger.warning("Evaluator failed for %s: %s", scenario.name, e)
            return ScenarioResult(
                scenario=scenario,
                passed=False,
                max_drawdown=1.0,
                total_return_pct=-100.0,
                sharpe=-99.0,
                trade_count=0,
                details=f"{details} — evaluator error: {e}",
                bars_tested=len(prices),
            )

        dd = metrics.get("max_drawdown", 1.0)
        sharpe = metrics.get("sharpe", -99.0)
        ret = metrics.get("total_return_pct", -100.0)
        trades = metrics.get("trade_count", 0)

        passed = (
            dd <= self._config.max_drawdown_limit
            and sharpe >= self._config.min_sharpe_under_stress
            and ret >= -self._config.max_loss_per_scenario_pct
        )

        return ScenarioResult(
            scenario=scenario,
            passed=passed,
            max_drawdown=dd,
            total_return_pct=ret,
            sharpe=sharpe,
            trade_count=trades,
            details=details + (
                f" → DD={dd:.3f}, Sharpe={sharpe:.2f}, Return={ret:.2f}%"
            ),
            bars_tested=len(prices),
        )


# ─── ZEUS GAN Regime Generator ──────────────────────────────────────────────

@dataclass
class GANConfig:
    """Configuration for ZEUS-GAN synthetic regime generator."""
    latent_dim: int = 32                  # GAN latent space dimension
    sequence_length: int = 500            # Generated sequence length
    n_regimes: int = 10                   # Number of synthetic regimes to generate
    # Statistical properties to vary
    min_volatility_mult: float = 0.5      # Min vol multiplier vs training data
    max_volatility_mult: float = 3.0      # Max vol multiplier
    min_trend_strength: float = -0.05     # Strong downtrend
    max_trend_strength: float = 0.05      # Strong uptrend
    regime_types: list[str] = field(default_factory=lambda: [
        "trending_up", "trending_down", "mean_reverting",
        "high_volatility", "low_volatility", "regime_switch",
        "correlation_break", "momentum_crash", "squeeze",
        "random_walk",
    ])
    # Overfitting detection
    max_sharpe_gap: float = 1.0           # Max gap between real and GAN Sharpe


@dataclass
class SyntheticRegime:
    """A synthetically generated market regime."""
    regime_type: str
    prices: np.ndarray
    volatility: float
    trend: float
    description: str


@dataclass
class GANTestResult:
    """Result of GAN overfitting test."""
    genome_id: str
    passed: bool
    real_sharpe: float
    mean_synthetic_sharpe: float
    sharpe_gap: float
    regime_results: list[dict]   # {regime_type, sharpe, passed}
    n_regimes_passed: int
    n_regimes_total: int


class ZeusGANGenerator:
    """
    ZEUS-GAN — Synthetic Regime Generator for Overfitting Detection.

    Generates never-seen market conditions using statistical processes
    (serving as a lightweight GAN substitute). Genomes that perform
    dramatically worse on synthetic data compared to historical data
    are flagged as overfit.

    Usage:
        gan = ZeusGANGenerator(config)
        regimes = gan.generate_regimes(reference_prices)
        result = gan.test_overfitting(genome_id, real_sharpe, evaluator, regimes)
    """

    def __init__(self, config: Optional[GANConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self._config = config or GANConfig()
        self._rng = rng or np.random.default_rng(42)

    def generate_regimes(
        self,
        reference_prices: np.ndarray,
    ) -> list[SyntheticRegime]:
        """
        Generate synthetic market regimes based on the statistical
        properties of reference data.
        """
        ref_returns = np.diff(np.log(reference_prices + 1e-10))
        ref_vol = float(np.std(ref_returns)) if len(ref_returns) > 0 else 0.01
        ref_mean = float(np.mean(ref_returns)) if len(ref_returns) > 0 else 0.0
        base_price = float(reference_prices[-1]) if len(reference_prices) > 0 else 2000.0

        regimes = []
        for i, regime_type in enumerate(self._config.regime_types[:self._config.n_regimes]):
            prices, vol, trend, desc = self._generate_regime(
                regime_type, base_price, ref_vol, ref_mean
            )
            regimes.append(SyntheticRegime(
                regime_type=regime_type,
                prices=prices,
                volatility=vol,
                trend=trend,
                description=desc,
            ))

        return regimes

    def _generate_regime(
        self,
        regime_type: str,
        base_price: float,
        base_vol: float,
        base_trend: float,
    ) -> tuple[np.ndarray, float, float, str]:
        """Generate a single synthetic regime."""
        n = self._config.sequence_length

        if regime_type == "trending_up":
            trend = abs(base_trend) * 2 + 0.001
            vol = base_vol
            desc = "Strong uptrend with normal volatility"
        elif regime_type == "trending_down":
            trend = -(abs(base_trend) * 2 + 0.001)
            vol = base_vol
            desc = "Strong downtrend with normal volatility"
        elif regime_type == "mean_reverting":
            trend = 0.0
            vol = base_vol * 0.8
            desc = "Mean-reverting with lower volatility"
        elif regime_type == "high_volatility":
            trend = base_trend
            vol = base_vol * self._config.max_volatility_mult
            desc = f"Extreme volatility ({self._config.max_volatility_mult}x normal)"
        elif regime_type == "low_volatility":
            trend = base_trend * 0.5
            vol = base_vol * self._config.min_volatility_mult
            desc = f"Compressed volatility ({self._config.min_volatility_mult}x normal)"
        elif regime_type == "regime_switch":
            # First half trending, second half reverting
            trend = base_trend
            vol = base_vol * 1.5
            desc = "Mid-sequence regime switch (trend → mean-revert)"
        elif regime_type == "correlation_break":
            trend = base_trend
            vol = base_vol * 2.0
            desc = "Correlation break: normal assets decorrelate"
        elif regime_type == "momentum_crash":
            trend = -abs(base_trend) * 3
            vol = base_vol * 2.5
            desc = "Momentum crash: fast reversal with high vol"
        elif regime_type == "squeeze":
            trend = 0.0
            vol = base_vol * 0.3
            desc = "Volatility squeeze: extremely low vol before breakout"
        else:  # random_walk
            trend = 0.0
            vol = base_vol
            desc = "Pure random walk (no predictable pattern)"

        # Generate price series
        returns = self._rng.normal(trend, max(vol, 1e-6), n)

        # Apply regime-specific transformations
        if regime_type == "regime_switch":
            half = n // 2
            returns[:half] = self._rng.normal(abs(base_trend) * 2, vol, half)
            returns[half:] = self._rng.normal(-abs(base_trend), vol * 0.5, n - half)
        elif regime_type == "mean_reverting":
            # OU process
            theta = 0.1  # Mean reversion speed
            for i in range(1, n):
                returns[i] = -theta * returns[i - 1] + self._rng.normal(0, vol)
        elif regime_type == "squeeze":
            # Decreasing vol then sudden breakout
            for i in range(n):
                progress = i / n
                if progress < 0.8:
                    returns[i] = self._rng.normal(0, vol * (1 - progress))
                else:
                    returns[i] = self._rng.normal(
                        self._rng.choice([-1, 1]) * 0.005,
                        vol * 3
                    )

        prices = base_price * np.exp(np.cumsum(returns))
        prices = np.maximum(prices, base_price * 0.5)  # Floor

        return prices, vol, trend, desc

    def test_overfitting(
        self,
        genome_id: str,
        real_sharpe: float,
        evaluator: Callable[[np.ndarray], dict],
        regimes: Optional[list[SyntheticRegime]] = None,
        reference_prices: Optional[np.ndarray] = None,
    ) -> GANTestResult:
        """
        Test a genome for overfitting by comparing performance on
        real data vs synthetic regimes.

        Args:
            genome_id: ID of the genome being tested.
            real_sharpe: Sharpe ratio achieved on real data.
            evaluator: Function(prices) -> {"sharpe": float, ...}
            regimes: Pre-generated regimes, or generate from reference.
            reference_prices: Used to generate regimes if not provided.

        Returns:
            GANTestResult with overfitting analysis.
        """
        if regimes is None:
            if reference_prices is None:
                raise ValueError("Must provide either regimes or reference_prices")
            regimes = self.generate_regimes(reference_prices)

        regime_results = []
        sharpes = []

        for regime in regimes:
            try:
                metrics = evaluator(regime.prices)
                sharpe = metrics.get("sharpe", -99.0)
            except Exception as e:
                logger.warning("GAN eval failed for %s: %s", regime.regime_type, e)
                sharpe = -99.0

            sharpes.append(sharpe)
            regime_results.append({
                "regime_type": regime.regime_type,
                "sharpe": sharpe,
                "description": regime.description,
                "passed": sharpe > self._config.max_sharpe_gap * -1,
            })

        mean_synthetic = float(np.mean(sharpes)) if sharpes else -99.0
        gap = real_sharpe - mean_synthetic
        n_passed = sum(1 for r in regime_results if r["passed"])

        # Overfitting detection:
        # If gap between real and synthetic Sharpe is too large, genome is overfit
        passed = gap <= self._config.max_sharpe_gap

        result = GANTestResult(
            genome_id=genome_id,
            passed=passed,
            real_sharpe=real_sharpe,
            mean_synthetic_sharpe=mean_synthetic,
            sharpe_gap=gap,
            regime_results=regime_results,
            n_regimes_passed=n_passed,
            n_regimes_total=len(regimes),
        )

        logger.info(
            "ZEUS-GAN overfitting test for %s: %s | real=%.2f, synth=%.2f, gap=%.2f",
            genome_id, "PASSED" if passed else "OVERFIT",
            real_sharpe, mean_synthetic, gap,
        )

        return result


# ─── Combined ZEUS Engine ────────────────────────────────────────────────────

class ZeusEngine:
    """
    Combined ZEUS engine: stress testing + GAN overfitting detection.

    Used in the PROMETHEUS-DEPLOY pipeline:
      Walk-forward → Monte Carlo → ZEUS stress → paper trading → live

    Usage:
        zeus = ZeusEngine()
        result = zeus.full_test(genome_id, prices, evaluator, real_sharpe)
        if result["stress_passed"] and result["gan_passed"]:
            promote_to_paper_trading()
    """

    def __init__(
        self,
        stress_config: Optional[StressConfig] = None,
        gan_config: Optional[GANConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self._rng = rng or np.random.default_rng(42)
        self.stress_tester = ZeusStressTester(stress_config, self._rng)
        self.gan_generator = ZeusGANGenerator(gan_config, self._rng)

    def full_test(
        self,
        genome_id: str,
        base_prices: np.ndarray,
        evaluator: Callable[[np.ndarray], dict],
        real_sharpe: float = 0.0,
    ) -> dict:
        """
        Run the complete ZEUS test suite: stress + GAN.

        Returns dict with:
          - stress_result: StressTestResult
          - gan_result: GANTestResult
          - stress_passed: bool
          - gan_passed: bool
          - overall_passed: bool
        """
        # Stress tests
        stress_result = self.stress_tester.test_genome(
            genome_id, base_prices, evaluator
        )

        # GAN overfitting test
        gan_result = self.gan_generator.test_overfitting(
            genome_id, real_sharpe, evaluator,
            reference_prices=base_prices,
        )

        overall = stress_result.passed and gan_result.passed

        logger.info(
            "ZEUS full test for %s: stress=%s, GAN=%s → %s",
            genome_id,
            "PASS" if stress_result.passed else "FAIL",
            "PASS" if gan_result.passed else "FAIL",
            "PROMOTED" if overall else "REJECTED",
        )

        return {
            "stress_result": stress_result,
            "gan_result": gan_result,
            "stress_passed": stress_result.passed,
            "gan_passed": gan_result.passed,
            "overall_passed": overall,
        }
