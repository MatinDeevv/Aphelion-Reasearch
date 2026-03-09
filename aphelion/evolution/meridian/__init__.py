"""
MERIDIAN — Dynamic Multi-Timeframe Weighting

Sub-components:
  MERIDIAN-GRANGER  – Rolling Granger causality F-statistics
  MERIDIAN-WEIGHTS  – Dynamic timeframe weight vector [w_1m, w_5m, w_15m, w_1h]
"""

from aphelion.evolution.meridian.engine import (
    GrangerResult,
    MeridianConfig,
    MeridianEngine,
    MeridianState,
    granger_causality_f,
)

__all__ = [
    "GrangerResult",
    "MeridianConfig",
    "MeridianEngine",
    "MeridianState",
    "granger_causality_f",
]
