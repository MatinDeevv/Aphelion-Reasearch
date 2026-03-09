"""
APHELION PROMETHEUS — Evolutionary Engine
Orchestrates the full genetic optimisation loop:
  1. Initialise random population
  2. Evaluate each genome via backtesting
  3. Rank by multi-objective fitness
  4. Evolve via NEAT operators
  5. Track hall-of-fame and generational statistics
  6. Save/load checkpoints

The engine is designed to be run offline with historical bar data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from aphelion.core.config import GENOME_DIR
from aphelion.evolution.prometheus.genome import (
    Genome, GenomeFitness, default_genome, random_genome, N_GENES,
)
from aphelion.evolution.prometheus.neat import (
    NEATConfig, assign_species, next_generation, elite_selection, Species,
)

logger = logging.getLogger(__name__)


# ─── Engine Configuration ────────────────────────────────────────────────────

@dataclass
class EvolutionConfig:
    """Top-level evolution engine settings."""
    neat: NEATConfig = field(default_factory=NEATConfig)
    max_generations: int = 100
    target_sharpe: float = 1.5             # Stop early if reached
    min_trades_per_genome: int = 30        # Require enough sample
    hall_of_fame_size: int = 10
    checkpoint_dir: str = str(Path(GENOME_DIR) / "live")
    seed: int = 42
    parallel_evaluations: int = 1          # Future: multiprocessing
    # Convergence detection
    convergence_window: int = 10           # Last N gens to check
    convergence_threshold: float = 0.01    # Min improvement per gen


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_composite: float
    mean_composite: float
    worst_composite: float
    best_sharpe: float
    mean_sharpe: float
    best_trade_count: int
    species_count: int
    elapsed_seconds: float


# ─── Main Engine ─────────────────────────────────────────────────────────────

class PrometheusEngine:
    """
    Evolutionary optimisation engine for APHELION trading strategies.

    Usage:
        engine = PrometheusEngine(config)
        engine.set_evaluator(my_backtest_fn)
        results = engine.run()
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self._config = config or EvolutionConfig()
        self._rng = np.random.default_rng(self._config.seed)

        self._population: list[Genome] = []
        self._species: list[Species] = []
        self._hall_of_fame: list[Genome] = []
        self._generation_stats: list[GenerationStats] = []
        self._current_generation: int = 0
        self._evaluator: Optional[Callable[[Genome], GenomeFitness]] = None

        self._best_ever: Optional[Genome] = None
        self._converged: bool = False

    # ── Evaluator ────────────────────────────────────────────────────────────

    def set_evaluator(self, fn: Callable[[Genome], GenomeFitness]) -> None:
        """
        Set the fitness evaluator function.

        The evaluator receives a Genome and must return a GenomeFitness.
        Typically this runs a full backtest with the genome's strategy config.
        """
        self._evaluator = fn

    # ── Initialise ───────────────────────────────────────────────────────────

    def initialise_population(self) -> None:
        """Create the initial random population, seeded with the default genome."""
        pop_size = self._config.neat.population_size

        self._population = [default_genome(generation=0)]  # Seed with known-good defaults

        while len(self._population) < pop_size:
            self._population.append(random_genome(self._rng, generation=0))

        self._current_generation = 0
        logger.info("Initialised population: %d genomes", len(self._population))

    # ── Evaluate ─────────────────────────────────────────────────────────────

    def evaluate_population(self) -> None:
        """Evaluate all genomes in the current population."""
        if self._evaluator is None:
            raise RuntimeError("No evaluator set. Call set_evaluator() first.")

        for i, genome in enumerate(self._population):
            try:
                fitness = self._evaluator(genome)
                genome.fitness = fitness
                fitness.compute_composite()
            except Exception:
                logger.warning(
                    "Evaluation failed for genome %s", genome.genome_id, exc_info=True
                )
                genome.fitness = GenomeFitness(composite=-999.0)

    # ── Single Generation Step ───────────────────────────────────────────────

    def step(self) -> GenerationStats:
        """Run one generation: evaluate → stats → speciate → evolve."""
        t0 = time.time()

        # 1. Evaluate
        self.evaluate_population()

        # 2. Collect stats
        composites = [g.fitness.composite for g in self._population]
        sharpes = [g.fitness.sharpe for g in self._population]
        trade_counts = [g.fitness.trade_count for g in self._population]

        stats = GenerationStats(
            generation=self._current_generation,
            best_composite=max(composites) if composites else 0.0,
            mean_composite=float(np.mean(composites)) if composites else 0.0,
            worst_composite=min(composites) if composites else 0.0,
            best_sharpe=max(sharpes) if sharpes else 0.0,
            mean_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
            best_trade_count=max(trade_counts) if trade_counts else 0,
            species_count=len(self._species),
            elapsed_seconds=0.0,
        )
        self._generation_stats.append(stats)

        # 3. Update hall of fame
        self._update_hall_of_fame()

        # 4. Update best-ever
        best_this_gen = max(self._population, key=lambda g: g.fitness.composite)
        if self._best_ever is None or best_this_gen.fitness.composite > self._best_ever.fitness.composite:
            self._best_ever = best_this_gen

        # 5. Speciate
        self._species = assign_species(
            self._population, self._species, self._config.neat
        )

        # 6. Evolve next generation
        self._current_generation += 1
        self._population = next_generation(
            self._population, self._config.neat,
            self._current_generation, self._rng,
        )

        stats.elapsed_seconds = time.time() - t0

        logger.info(
            "Gen %d | best=%.4f mean=%.4f sharpe=%.3f species=%d (%.1fs)",
            stats.generation, stats.best_composite, stats.mean_composite,
            stats.best_sharpe, stats.species_count, stats.elapsed_seconds,
        )

        return stats

    # ── Full Run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the full evolutionary loop.

        Returns dict with:
          - best_genome: The all-time best Genome
          - hall_of_fame: Top genomes across all generations
          - stats: List of GenerationStats
          - converged: Whether convergence was detected
          - total_generations: How many generations ran
        """
        if not self._population:
            self.initialise_population()

        for gen in range(self._config.max_generations):
            stats = self.step()

            # Early stop: target Sharpe reached
            if stats.best_sharpe >= self._config.target_sharpe:
                logger.info(
                    "Target Sharpe %.2f reached at gen %d",
                    self._config.target_sharpe, gen,
                )
                break

            # Convergence detection
            if self._check_convergence():
                self._converged = True
                logger.info("Convergence detected at gen %d", gen)
                break

            # Checkpoint
            if gen % 10 == 0 and gen > 0:
                self.save_checkpoint()

        self.save_checkpoint()

        return {
            "best_genome": self._best_ever,
            "hall_of_fame": list(self._hall_of_fame),
            "stats": list(self._generation_stats),
            "converged": self._converged,
            "total_generations": self._current_generation,
        }

    # ── Hall of Fame ─────────────────────────────────────────────────────────

    def _update_hall_of_fame(self) -> None:
        """Merge current population's top performers into the hall of fame."""
        candidates = list(self._hall_of_fame) + list(self._population)
        # Deduplicate by genome_id
        seen = set()
        unique = []
        for g in candidates:
            if g.genome_id not in seen:
                seen.add(g.genome_id)
                unique.append(g)
        ranked = sorted(unique, key=lambda g: g.fitness.composite, reverse=True)
        self._hall_of_fame = ranked[: self._config.hall_of_fame_size]

    # ── Convergence ──────────────────────────────────────────────────────────

    def _check_convergence(self) -> bool:
        window = self._config.convergence_window
        if len(self._generation_stats) < window:
            return False
        recent = [s.best_composite for s in self._generation_stats[-window:]]
        improvement = max(recent) - min(recent)
        return improvement < self._config.convergence_threshold

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save_checkpoint(self, directory: Optional[str] = None) -> Path:
        """Save current state to disk."""
        ckpt_dir = Path(directory or self._config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "generation": self._current_generation,
            "population": [g.to_dict() for g in self._population],
            "hall_of_fame": [g.to_dict() for g in self._hall_of_fame],
            "best_ever": self._best_ever.to_dict() if self._best_ever else None,
            "stats": [
                {
                    "generation": s.generation,
                    "best_composite": s.best_composite,
                    "mean_composite": s.mean_composite,
                    "best_sharpe": s.best_sharpe,
                    "mean_sharpe": s.mean_sharpe,
                    "species_count": s.species_count,
                }
                for s in self._generation_stats
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        path = ckpt_dir / f"prometheus_gen{self._current_generation:04d}.json"
        path.write_text(json.dumps(state, indent=2))
        logger.info("Checkpoint saved: %s", path)

        # Also save the hall of fame separately
        hof_path = ckpt_dir / "hall_of_fame.json"
        hof_path.write_text(
            json.dumps([g.to_dict() for g in self._hall_of_fame], indent=2)
        )

        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load state from a checkpoint file."""
        data = json.loads(Path(path).read_text())
        self._current_generation = data["generation"]
        self._population = [Genome.from_dict(d) for d in data["population"]]
        self._hall_of_fame = [Genome.from_dict(d) for d in data.get("hall_of_fame", [])]
        if data.get("best_ever"):
            self._best_ever = Genome.from_dict(data["best_ever"])
        logger.info(
            "Loaded checkpoint: gen %d, %d genomes",
            self._current_generation, len(self._population),
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def population(self) -> list[Genome]:
        return self._population

    @property
    def best_genome(self) -> Optional[Genome]:
        return self._best_ever

    @property
    def hall_of_fame(self) -> list[Genome]:
        return list(self._hall_of_fame)

    @property
    def generation(self) -> int:
        return self._current_generation

    @property
    def stats(self) -> list[GenerationStats]:
        return list(self._generation_stats)
