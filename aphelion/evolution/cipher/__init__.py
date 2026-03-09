"""
CIPHER — Feature Importance & Alpha Decay Detection

Sub-components:
  CIPHER-DECAY     – Rolling SHAP importance tracker (30d vs 90d ratio)
  CIPHER-HALFLIFE  – Exponential decay curve fitting, half-life estimation
"""

from aphelion.evolution.cipher.engine import (
    CipherConfig,
    CipherEngine,
    DecayAlert,
    FeatureImportance,
    PermutationImportanceComputer,
    estimate_half_life,
)

__all__ = [
    "CipherConfig",
    "CipherEngine",
    "DecayAlert",
    "FeatureImportance",
    "PermutationImportanceComputer",
    "estimate_half_life",
]
