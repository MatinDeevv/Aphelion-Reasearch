"""
APHELION PROMETHEUS — NEAT Evolutionary Strategy Optimisation (Phase 8)

Modules:
  genome    — Strategy genome encoding (gene vector + fitness)
  neat      — NEAT operators (mutation, crossover, selection, speciation)
  engine    — Full evolutionary loop with checkpointing
  evaluator — Genome → backtest → fitness bridge
"""

from aphelion.evolution.prometheus.genome import (
    GENE_SPEC,
    GENE_NAMES,
    N_GENES,
    Genome,
    GenomeFitness,
    random_genome,
    default_genome,
)
from aphelion.evolution.prometheus.neat import (
    NEATConfig,
    Species,
    mutate,
    crossover_uniform,
    crossover_blend,
    tournament_select,
    elite_selection,
    genome_distance,
    assign_species,
    next_generation,
)
from aphelion.evolution.prometheus.engine import (
    EvolutionConfig,
    GenerationStats,
    PrometheusEngine,
)
from aphelion.evolution.prometheus.evaluator import (
    EvaluatorConfig,
    GenomeStrategy,
    evaluate_genome,
)

__all__ = [
    # Genome
    "GENE_SPEC", "GENE_NAMES", "N_GENES",
    "Genome", "GenomeFitness", "random_genome", "default_genome",
    # NEAT
    "NEATConfig", "Species",
    "mutate", "crossover_uniform", "crossover_blend",
    "tournament_select", "elite_selection",
    "genome_distance", "assign_species", "next_generation",
    # Engine
    "EvolutionConfig", "GenerationStats", "PrometheusEngine",
    # Evaluator
    "EvaluatorConfig", "GenomeStrategy", "evaluate_genome",
]
