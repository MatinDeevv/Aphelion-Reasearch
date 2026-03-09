"""
APHELION PROMETHEUS — Strategy Genome
Encodes a trading strategy as a numeric genome that can be mutated,
crossed-over, and evaluated by the evolutionary engine.

Each gene controls a strategy parameter:
  - Entry/exit thresholds, indicator weights, timeframe preferences
  - Risk parameters (SL/TP multipliers, position sizing)
  - Regime-adaptive behaviour toggles
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Gene Definitions ────────────────────────────────────────────────────────
# Each gene has (name, min, max, default, dtype)

GENE_SPEC: list[tuple[str, float, float, float, str]] = [
    # ── Entry ────────────────────────────────────────────────────────────────
    ("confidence_threshold",    0.40, 0.90, 0.55, "float"),
    ("horizon_agreement_min",   0.33, 1.00, 0.66, "float"),
    ("uncertainty_ceiling",     0.30, 0.95, 0.80, "float"),
    # ── Risk ─────────────────────────────────────────────────────────────────
    ("atr_sl_multiplier",       1.0,  5.0,  2.0,  "float"),
    ("rr_ratio",                1.5,  5.0,  2.0,  "float"),
    ("risk_per_trade",          0.005, 0.02, 0.015, "float"),
    ("kelly_fraction",          0.10, 0.50, 0.25, "float"),
    ("use_kelly",               0.0,  1.0,  1.0,  "bool"),
    # ── Timing ───────────────────────────────────────────────────────────────
    ("cooldown_bars",           1.0,  20.0, 5.0,  "int"),
    ("max_open_positions",      1.0,  3.0,  3.0,  "int"),
    # ── Regime adaptation ────────────────────────────────────────────────────
    ("trend_bonus",             0.00, 0.15, 0.05, "float"),
    ("range_penalty",           0.00, 0.15, 0.03, "float"),
    ("vol_exp_penalty",         0.00, 0.20, 0.10, "float"),
    # ── Feature weighting ────────────────────────────────────────────────────
    ("weight_vpin",             0.0,  2.0,  1.0,  "float"),
    ("weight_ofi",              0.0,  2.0,  1.0,  "float"),
    ("weight_vwap_dist",        0.0,  2.0,  1.0,  "float"),
    ("weight_atr",              0.0,  2.0,  1.0,  "float"),
    ("weight_rsi",              0.0,  2.0,  1.0,  "float"),
    ("weight_spread",           0.0,  2.0,  1.0,  "float"),
    # ── Session filters (0=disallow, 1=allow) ────────────────────────────────
    ("allow_asian",             0.0,  1.0,  1.0,  "bool"),
    ("allow_london",            0.0,  1.0,  1.0,  "bool"),
    ("allow_new_york",          0.0,  1.0,  1.0,  "bool"),
    ("allow_overlap",           0.0,  1.0,  1.0,  "bool"),
    ("allow_dead_zone",         0.0,  1.0,  0.0,  "bool"),
]

N_GENES = len(GENE_SPEC)
GENE_NAMES = [g[0] for g in GENE_SPEC]


@dataclass
class GenomeFitness:
    """Multi-objective fitness scores from backtest evaluation."""
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 1.0
    win_rate: float = 0.0
    trade_count: int = 0
    expectancy: float = 0.0
    dsr: float = 0.0                       # Deflated Sharpe
    # Composite fitness (used for ranking)
    composite: float = 0.0

    def compute_composite(self) -> float:
        """
        Multi-objective composite: weighted combination of key metrics.
        Penalises low trade counts and extreme drawdowns.
        """
        if self.trade_count < 10:
            self.composite = -999.0
            return self.composite

        score = (
            self.sharpe * 0.30
            + self.sortino * 0.15
            + self.calmar * 0.10
            + self.profit_factor * 0.10
            + self.expectancy * 0.10
            + self.win_rate * 0.05
            + self.dsr * 0.20
        )
        # Drawdown penalty: heavily penalise >20% drawdown
        if self.max_drawdown > 0.20:
            score *= 0.5
        if self.max_drawdown > 0.40:
            score *= 0.1

        self.composite = score
        return self.composite


@dataclass
class Genome:
    """
    A single strategy genome: a float vector encoding strategy parameters.
    """
    genes: np.ndarray                       # shape (N_GENES,)
    genome_id: str = ""
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    fitness: GenomeFitness = field(default_factory=GenomeFitness)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.genome_id:
            self.genome_id = self._make_id()

    def _make_id(self) -> str:
        h = hashlib.sha256(self.genes.tobytes()).hexdigest()[:12]
        return f"G-{self.generation:04d}-{h}"

    # ── Gene access by name ──────────────────────────────────────────────────

    def get(self, name: str) -> float:
        idx = GENE_NAMES.index(name)
        spec = GENE_SPEC[idx]
        raw = float(self.genes[idx])
        if spec[4] == "bool":
            return 1.0 if raw >= 0.5 else 0.0
        if spec[4] == "int":
            return float(round(raw))
        return raw

    def to_strategy_config(self) -> dict:
        """Convert genome to a strategy configuration dict."""
        return {name: self.get(name) for name in GENE_NAMES}

    # ── Clamp to legal bounds ────────────────────────────────────────────────

    def clamp(self) -> None:
        """Enforce gene bounds from GENE_SPEC."""
        for i, (_, lo, hi, _, _) in enumerate(GENE_SPEC):
            self.genes[i] = np.clip(self.genes[i], lo, hi)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "genes": self.genes.tolist(),
            "parent_ids": self.parent_ids,
            "fitness": {
                "sharpe": self.fitness.sharpe,
                "sortino": self.fitness.sortino,
                "calmar": self.fitness.calmar,
                "profit_factor": self.fitness.profit_factor,
                "total_return_pct": self.fitness.total_return_pct,
                "max_drawdown": self.fitness.max_drawdown,
                "win_rate": self.fitness.win_rate,
                "trade_count": self.fitness.trade_count,
                "expectancy": self.fitness.expectancy,
                "dsr": self.fitness.dsr,
                "composite": self.fitness.composite,
            },
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        fitness_d = d.get("fitness", {})
        return cls(
            genes=np.array(d["genes"], dtype=np.float64),
            genome_id=d["genome_id"],
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
            fitness=GenomeFitness(**fitness_d),
            created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(timezone.utc),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Saved genome %s → %s", self.genome_id, path)

    @classmethod
    def load(cls, path: Path) -> Genome:
        return cls.from_dict(json.loads(path.read_text()))


# ─── Factory Functions ───────────────────────────────────────────────────────

def random_genome(rng: Optional[np.random.Generator] = None, generation: int = 0) -> Genome:
    """Create a random genome with genes uniformly sampled in valid bounds."""
    rng = rng or np.random.default_rng()
    genes = np.empty(N_GENES, dtype=np.float64)
    for i, (_, lo, hi, _, _) in enumerate(GENE_SPEC):
        genes[i] = rng.uniform(lo, hi)
    g = Genome(genes=genes, generation=generation)
    g.clamp()
    return g


def default_genome(generation: int = 0) -> Genome:
    """Create a genome with all genes at their spec defaults."""
    genes = np.array([g[3] for g in GENE_SPEC], dtype=np.float64)
    return Genome(genes=genes, generation=generation)
