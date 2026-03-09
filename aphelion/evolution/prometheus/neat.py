"""
APHELION PROMETHEUS — NEAT Operators
Crossover, mutation, selection, and speciation operators for
NeuroEvolution of Augmenting Topologies applied to strategy genomes.

Supports:
  - Gaussian mutation with adaptive step sizes
  - Uniform / blend crossover
  - Tournament + elite selection
  - Speciation via genome distance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from aphelion.evolution.prometheus.genome import (
    GENE_SPEC, N_GENES, Genome, GenomeFitness, random_genome,
)

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class NEATConfig:
    """Configuration for the NEAT evolutionary operators."""
    # Population
    population_size: int = 50
    elite_count: int = 5                    # Top N survive untouched
    # Mutation
    mutation_rate: float = 0.30             # Probability gene mutates
    mutation_sigma: float = 0.10            # Gaussian σ (fraction of range)
    mutation_sigma_decay: float = 0.995     # Decay per generation
    reset_gene_probability: float = 0.02    # Chance gene resets to random
    # Crossover
    crossover_rate: float = 0.60            # Probability of crossover vs clone
    blend_alpha: float = 0.3               # BLX-α blending factor
    # Selection
    tournament_size: int = 5
    # Speciation
    compatibility_threshold: float = 3.0
    excess_coefficient: float = 1.0
    disjoint_coefficient: float = 1.0
    weight_coefficient: float = 0.4
    # Stagnation
    stagnation_limit: int = 15              # Generations without improvement → purge


# ─── Mutation ────────────────────────────────────────────────────────────────

def mutate(
    genome: Genome,
    config: NEATConfig,
    rng: Optional[np.random.Generator] = None,
    generation: int = 0,
) -> Genome:
    """
    Apply Gaussian mutation to a genome copy.
    Each gene has `mutation_rate` probability of being perturbed.
    """
    rng = rng or np.random.default_rng()
    new_genes = genome.genes.copy()

    for i, (_, lo, hi, default, _) in enumerate(GENE_SPEC):
        if rng.random() < config.mutation_rate:
            gene_range = hi - lo
            sigma = config.mutation_sigma * gene_range

            if rng.random() < config.reset_gene_probability:
                # Full reset to random value
                new_genes[i] = rng.uniform(lo, hi)
            else:
                new_genes[i] += rng.normal(0, sigma)

            new_genes[i] = np.clip(new_genes[i], lo, hi)

    child = Genome(
        genes=new_genes,
        generation=generation,
        parent_ids=[genome.genome_id],
    )
    child.clamp()
    return child


# ─── Crossover ───────────────────────────────────────────────────────────────

def crossover_uniform(
    parent_a: Genome,
    parent_b: Genome,
    rng: Optional[np.random.Generator] = None,
    generation: int = 0,
) -> Genome:
    """Uniform crossover: each gene randomly picked from either parent."""
    rng = rng or np.random.default_rng()
    mask = rng.random(N_GENES) < 0.5
    genes = np.where(mask, parent_a.genes, parent_b.genes)
    child = Genome(
        genes=genes,
        generation=generation,
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
    )
    child.clamp()
    return child


def crossover_blend(
    parent_a: Genome,
    parent_b: Genome,
    alpha: float = 0.3,
    rng: Optional[np.random.Generator] = None,
    generation: int = 0,
) -> Genome:
    """
    BLX-α crossover: child gene in [min-α*d, max+α*d] where d = |a-b|.
    Produces more diverse offspring than uniform crossover.
    """
    rng = rng or np.random.default_rng()
    genes = np.empty(N_GENES, dtype=np.float64)

    for i in range(N_GENES):
        a, b = parent_a.genes[i], parent_b.genes[i]
        lo_gene, hi_gene = min(a, b), max(a, b)
        d = hi_gene - lo_gene
        genes[i] = rng.uniform(lo_gene - alpha * d, hi_gene + alpha * d)

    child = Genome(
        genes=genes,
        generation=generation,
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
    )
    child.clamp()
    return child


# ─── Selection ───────────────────────────────────────────────────────────────

def tournament_select(
    population: list[Genome],
    k: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Genome:
    """Tournament selection: pick k random, return best composite fitness."""
    rng = rng or np.random.default_rng()
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    candidates = [population[i] for i in indices]
    return max(candidates, key=lambda g: g.fitness.composite)


def elite_selection(population: list[Genome], n: int) -> list[Genome]:
    """Return top-n genomes by composite fitness (deterministic)."""
    ranked = sorted(population, key=lambda g: g.fitness.composite, reverse=True)
    return ranked[:n]


# ─── Speciation ──────────────────────────────────────────────────────────────

def genome_distance(a: Genome, b: Genome, config: NEATConfig) -> float:
    """
    Compute compatibility distance between two genomes.
    Uses weighted gene-value distance (simplified NEAT distance).
    """
    diff = np.abs(a.genes - b.genes)
    # Normalise each gene by its range
    ranges = np.array([hi - lo for _, lo, hi, _, _ in GENE_SPEC], dtype=np.float64)
    ranges = np.maximum(ranges, 1e-10)
    normed = diff / ranges
    return float(config.weight_coefficient * np.mean(normed) * N_GENES)


@dataclass
class Species:
    """A cluster of genetically similar genomes."""
    species_id: int
    representative: Genome
    members: list[Genome]
    best_fitness: float = -float("inf")
    stagnation_counter: int = 0

    @property
    def size(self) -> int:
        return len(self.members)


def assign_species(
    population: list[Genome],
    existing_species: list[Species],
    config: NEATConfig,
) -> list[Species]:
    """Assign each genome to the closest species or create a new one."""
    next_id = max((s.species_id for s in existing_species), default=0) + 1

    # Clear members
    for sp in existing_species:
        sp.members = []

    for genome in population:
        placed = False
        for sp in existing_species:
            dist = genome_distance(genome, sp.representative, config)
            if dist < config.compatibility_threshold:
                sp.members.append(genome)
                placed = True
                break
        if not placed:
            new_sp = Species(
                species_id=next_id,
                representative=genome,
                members=[genome],
            )
            existing_species.append(new_sp)
            next_id += 1

    # Remove empty species
    existing_species = [sp for sp in existing_species if sp.members]

    # Update representatives and stagnation
    for sp in existing_species:
        best = max(sp.members, key=lambda g: g.fitness.composite)
        if best.fitness.composite > sp.best_fitness:
            sp.best_fitness = best.fitness.composite
            sp.stagnation_counter = 0
            sp.representative = best
        else:
            sp.stagnation_counter += 1

    return existing_species


# ─── Full Generation Step ────────────────────────────────────────────────────

def next_generation(
    population: list[Genome],
    config: NEATConfig,
    generation: int,
    rng: Optional[np.random.Generator] = None,
) -> list[Genome]:
    """
    Produce the next generation from the current one.

    Steps:
      1. Elite survival (top N copied unchanged)
      2. Tournament selection → crossover or clone → mutation
      3. Fill to population_size
    """
    rng = rng or np.random.default_rng()

    if not population:
        return [random_genome(rng, generation) for _ in range(config.population_size)]

    # Sort by composite fitness
    ranked = sorted(population, key=lambda g: g.fitness.composite, reverse=True)

    new_pop: list[Genome] = []

    # 1. Elites survive unchanged
    for elite in ranked[: config.elite_count]:
        clone = Genome(
            genes=elite.genes.copy(),
            generation=generation,
            parent_ids=[elite.genome_id],
            metadata={"origin": "elite"},
        )
        clone.clamp()
        new_pop.append(clone)

    # 2. Fill remaining slots
    while len(new_pop) < config.population_size:
        if rng.random() < config.crossover_rate and len(ranked) >= 2:
            pa = tournament_select(ranked, config.tournament_size, rng)
            pb = tournament_select(ranked, config.tournament_size, rng)
            # Avoid self-cross
            attempts = 0
            while pb.genome_id == pa.genome_id and attempts < 5:
                pb = tournament_select(ranked, config.tournament_size, rng)
                attempts += 1

            if rng.random() < 0.5:
                child = crossover_uniform(pa, pb, rng, generation)
            else:
                child = crossover_blend(pa, pb, config.blend_alpha, rng, generation)
        else:
            parent = tournament_select(ranked, config.tournament_size, rng)
            child = Genome(
                genes=parent.genes.copy(),
                generation=generation,
                parent_ids=[parent.genome_id],
                metadata={"origin": "clone"},
            )
            child.clamp()

        child = mutate(child, config, rng, generation)
        new_pop.append(child)

    return new_pop[: config.population_size]
