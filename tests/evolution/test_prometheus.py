"""
Phase 8 Tests — PROMETHEUS NEAT Evolutionary Engine

Covers:
  - Genome creation, serialisation, gene access
  - NEAT operators: mutation, crossover, selection, speciation
  - Engine: initialisation, step, convergence, checkpointing
  - Evaluator: GenomeStrategy
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.evolution.prometheus.genome import (
    GENE_SPEC, GENE_NAMES, N_GENES,
    Genome, GenomeFitness, random_genome, default_genome,
)
from aphelion.evolution.prometheus.neat import (
    NEATConfig,
    mutate, crossover_uniform, crossover_blend,
    tournament_select, elite_selection,
    genome_distance, assign_species, next_generation,
)
from aphelion.evolution.prometheus.engine import (
    EvolutionConfig, PrometheusEngine,
)
from aphelion.evolution.prometheus.evaluator import GenomeStrategy


# ═══════════════════════════════════════════════════════════════════════════
# Genome
# ═══════════════════════════════════════════════════════════════════════════

class TestGenome:
    def test_gene_count(self):
        assert N_GENES == len(GENE_SPEC)
        assert N_GENES == len(GENE_NAMES)
        assert N_GENES > 15

    def test_default_genome_values(self):
        g = default_genome()
        assert g.genes.shape == (N_GENES,)
        assert g.get("confidence_threshold") == pytest.approx(0.55)
        assert g.get("rr_ratio") == pytest.approx(2.0)

    def test_random_genome_in_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            g = random_genome(rng)
            for i, (_, lo, hi, _, _) in enumerate(GENE_SPEC):
                assert lo <= g.genes[i] <= hi

    def test_genome_id_unique(self):
        g1 = random_genome(np.random.default_rng(1))
        g2 = random_genome(np.random.default_rng(2))
        assert g1.genome_id != g2.genome_id

    def test_genome_serialisation_round_trip(self):
        g = random_genome(np.random.default_rng(42))
        g.fitness = GenomeFitness(sharpe=1.5, win_rate=0.65, trade_count=50)
        d = g.to_dict()
        g2 = Genome.from_dict(d)
        assert np.allclose(g.genes, g2.genes)
        assert g2.fitness.sharpe == pytest.approx(1.5)

    def test_genome_save_load(self, tmp_path):
        g = default_genome()
        g.fitness.sharpe = 2.0
        path = tmp_path / "test_genome.json"
        g.save(path)
        g2 = Genome.load(path)
        assert np.allclose(g.genes, g2.genes)

    def test_to_strategy_config(self):
        g = default_genome()
        cfg = g.to_strategy_config()
        assert isinstance(cfg, dict)
        assert len(cfg) == N_GENES
        assert "confidence_threshold" in cfg

    def test_clamp_enforces_bounds(self):
        genes = np.full(N_GENES, 999.0)
        g = Genome(genes=genes)
        g.clamp()
        for i, (_, lo, hi, _, _) in enumerate(GENE_SPEC):
            assert g.genes[i] <= hi
            assert g.genes[i] >= lo

    def test_bool_gene_access(self):
        g = default_genome()
        assert g.get("use_kelly") == 1.0
        g.genes[GENE_NAMES.index("use_kelly")] = 0.3
        assert g.get("use_kelly") == 0.0

    def test_int_gene_access(self):
        g = default_genome()
        g.genes[GENE_NAMES.index("cooldown_bars")] = 7.6
        assert g.get("cooldown_bars") == 8.0


class TestGenomeFitness:
    def test_composite_requires_min_trades(self):
        f = GenomeFitness(sharpe=2.0, trade_count=5)
        f.compute_composite()
        assert f.composite == -999.0

    def test_composite_positive_for_good_strategy(self):
        f = GenomeFitness(
            sharpe=1.5, sortino=2.0, calmar=1.0,
            profit_factor=1.8, expectancy=0.02,
            win_rate=0.6, dsr=0.8, max_drawdown=0.10,
            trade_count=50,
        )
        f.compute_composite()
        assert f.composite > 0

    def test_high_drawdown_penalty(self):
        f1 = GenomeFitness(sharpe=1.5, sortino=1.0, max_drawdown=0.05, trade_count=50, dsr=0.5)
        f2 = GenomeFitness(sharpe=1.5, sortino=1.0, max_drawdown=0.30, trade_count=50, dsr=0.5)
        f1.compute_composite()
        f2.compute_composite()
        assert f1.composite > f2.composite


# ═══════════════════════════════════════════════════════════════════════════
# NEAT Operators
# ═══════════════════════════════════════════════════════════════════════════

class TestNEATOperators:
    def test_mutate_changes_genes(self):
        rng = np.random.default_rng(42)
        g = default_genome()
        cfg = NEATConfig(mutation_rate=1.0, mutation_sigma=0.5)
        m = mutate(g, cfg, rng)
        assert not np.allclose(g.genes, m.genes)
        assert m.parent_ids == [g.genome_id]

    def test_mutate_respects_bounds(self):
        rng = np.random.default_rng(42)
        cfg = NEATConfig(mutation_rate=1.0, mutation_sigma=1.0)
        for _ in range(50):
            g = random_genome(rng)
            m = mutate(g, cfg, rng)
            for i, (_, lo, hi, _, _) in enumerate(GENE_SPEC):
                assert lo <= m.genes[i] <= hi

    def test_crossover_uniform_inherits_from_both(self):
        rng = np.random.default_rng(42)
        a = random_genome(rng, generation=0)
        b = random_genome(rng, generation=0)
        child = crossover_uniform(a, b, rng)
        from_a = np.sum(np.isclose(child.genes, a.genes))
        from_b = np.sum(np.isclose(child.genes, b.genes))
        assert from_a > 0
        assert from_b > 0
        assert len(child.parent_ids) == 2

    def test_crossover_blend_produces_intermediate(self):
        rng = np.random.default_rng(42)
        a = default_genome()
        b = random_genome(rng)
        child = crossover_blend(a, b, alpha=0.0, rng=rng)
        for i in range(N_GENES):
            lo = min(a.genes[i], b.genes[i])
            hi = max(a.genes[i], b.genes[i])
            assert child.genes[i] >= lo - 0.01
            assert child.genes[i] <= hi + 0.01

    def test_tournament_select_picks_best(self):
        rng = np.random.default_rng(42)
        pop = [random_genome(rng) for _ in range(20)]
        for i, g in enumerate(pop):
            g.fitness.composite = float(i)
        winner = tournament_select(pop, k=20, rng=rng)
        assert winner.fitness.composite == 19.0

    def test_elite_selection(self):
        rng = np.random.default_rng(42)
        pop = [random_genome(rng) for _ in range(20)]
        for i, g in enumerate(pop):
            g.fitness.composite = float(i)
        elites = elite_selection(pop, 3)
        assert len(elites) == 3
        assert elites[0].fitness.composite == 19.0

    def test_genome_distance_self_zero(self):
        g = default_genome()
        cfg = NEATConfig()
        assert genome_distance(g, g, cfg) == pytest.approx(0.0)

    def test_genome_distance_positive_for_different(self):
        rng = np.random.default_rng(42)
        a = default_genome()
        b = random_genome(rng)
        cfg = NEATConfig()
        assert genome_distance(a, b, cfg) > 0

    def test_assign_species_creates_groups(self):
        rng = np.random.default_rng(42)
        pop = [random_genome(rng) for _ in range(20)]
        cfg = NEATConfig(compatibility_threshold=2.0)
        species = assign_species(pop, [], cfg)
        assert len(species) >= 1
        total_members = sum(s.size for s in species)
        assert total_members == 20

    def test_next_generation_correct_size(self):
        rng = np.random.default_rng(42)
        cfg = NEATConfig(population_size=30, elite_count=3)
        pop = [random_genome(rng) for _ in range(30)]
        for g in pop:
            g.fitness.composite = float(rng.random())
        new_pop = next_generation(pop, cfg, generation=1, rng=rng)
        assert len(new_pop) == 30


# ═══════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════

def _dummy_evaluator(genome: Genome) -> GenomeFitness:
    """Deterministic evaluator: fitness = mean of genes."""
    score = float(np.mean(genome.genes))
    return GenomeFitness(
        sharpe=score, sortino=score, calmar=score,
        profit_factor=max(score, 0.1),
        win_rate=0.5 + score * 0.01,
        trade_count=50,
        expectancy=score * 0.01,
        dsr=min(abs(score), 1.0),
        max_drawdown=0.05,
    )


class TestPrometheusEngine:
    def test_initialise_population(self):
        engine = PrometheusEngine(EvolutionConfig(
            neat=NEATConfig(population_size=10)))
        engine.initialise_population()
        assert len(engine.population) == 10

    def test_step_produces_stats(self):
        engine = PrometheusEngine(EvolutionConfig(
            neat=NEATConfig(population_size=10)))
        engine.set_evaluator(_dummy_evaluator)
        engine.initialise_population()
        stats = engine.step()
        assert stats.generation == 0
        assert stats.elapsed_seconds >= 0

    def test_run_returns_result(self):
        cfg = EvolutionConfig(
            neat=NEATConfig(population_size=10, elite_count=2),
            max_generations=3,
        )
        engine = PrometheusEngine(cfg)
        engine.set_evaluator(_dummy_evaluator)
        result = engine.run()
        assert result["best_genome"] is not None
        assert result["total_generations"] >= 1
        assert len(result["hall_of_fame"]) > 0

    def test_checkpoint_save_load(self, tmp_path):
        cfg = EvolutionConfig(
            neat=NEATConfig(population_size=10),
            checkpoint_dir=str(tmp_path),
        )
        engine = PrometheusEngine(cfg)
        engine.set_evaluator(_dummy_evaluator)
        engine.initialise_population()
        engine.step()
        ckpt = engine.save_checkpoint()
        assert ckpt.exists()

        engine2 = PrometheusEngine(cfg)
        engine2.load_checkpoint(ckpt)
        assert len(engine2.population) == 10

    def test_no_evaluator_raises(self):
        engine = PrometheusEngine()
        engine.initialise_population()
        with pytest.raises(RuntimeError, match="No evaluator"):
            engine.evaluate_population()

    def test_hall_of_fame_bounded(self):
        cfg = EvolutionConfig(
            neat=NEATConfig(population_size=20),
            max_generations=5,
            hall_of_fame_size=3,
        )
        engine = PrometheusEngine(cfg)
        engine.set_evaluator(_dummy_evaluator)
        engine.run()
        assert len(engine.hall_of_fame) <= 3


# ═══════════════════════════════════════════════════════════════════════════
# GenomeStrategy
# ═══════════════════════════════════════════════════════════════════════════

class TestGenomeStrategy:
    def test_strategy_callable(self):
        g = default_genome()
        strat = GenomeStrategy(g)
        assert callable(strat)
