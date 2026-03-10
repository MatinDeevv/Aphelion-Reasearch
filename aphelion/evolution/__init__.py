"""
APHELION Evolution Package

Sub-packages:
  cipher     – Feature importance & alpha decay detection (CIPHER-DECAY, CIPHER-HALFLIFE)
  meridian   – Dynamic multi-timeframe weighting (MERIDIAN-GRANGER, MERIDIAN-WEIGHTS)
  prometheus – Evolutionary engine (NEAT, PBT, Bayesian optimisation)
  zeus       – Pre-deployment stress testing & GAN overfitting detection
"""

from .auto_optimizer import (
    AutoOptimizer,
    PerformanceMonitor as OptPerformanceMonitor,
    DegradationSignal,
    OptAction,
    OptimizationRun,
)

__all__ = [
    "AutoOptimizer",
    "OptPerformanceMonitor",
    "DegradationSignal",
    "OptAction",
    "OptimizationRun",
]
