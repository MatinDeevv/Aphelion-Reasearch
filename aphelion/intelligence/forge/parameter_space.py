"""
FORGE — Parameter Space Definition
Phase 13 — Engineering Spec v3.0

Defines the optimizable parameter spaces for all strategies and sub-systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from aphelion.intelligence.forge.optimizer import ParameterSpec


class ParameterSpace:
    """
    Defines and manages the full parameter space for FORGE optimization.
    Each strategy/module registers its tunable parameters here.
    """

    def __init__(self):
        self._spaces: Dict[str, List[ParameterSpec]] = {}

    def register_module(self, module_name: str, params: List[ParameterSpec]) -> None:
        """Register a module's optimizable parameters."""
        self._spaces[module_name] = params

    def get_module_params(self, module_name: str) -> List[ParameterSpec]:
        return self._spaces.get(module_name, [])

    def get_all_params(self) -> List[ParameterSpec]:
        result = []
        for params in self._spaces.values():
            result.extend(params)
        return result

    @property
    def module_names(self) -> List[str]:
        return list(self._spaces.keys())

    @property
    def total_dimensions(self) -> int:
        return sum(len(p) for p in self._spaces.values())


def create_default_parameter_space() -> ParameterSpace:
    """Create the default parameter space per engineering spec."""
    space = ParameterSpace()

    # ALPHA M1 Scalping parameters
    space.register_module("ALPHA", [
        ParameterSpec("alpha_rsi_entry_low", 20.0, 40.0, 30.0, 1.0),
        ParameterSpec("alpha_rsi_entry_high", 60.0, 80.0, 70.0, 1.0),
        ParameterSpec("alpha_atr_multiplier_sl", 1.0, 3.0, 1.5, 0.1),
        ParameterSpec("alpha_atr_multiplier_tp", 1.5, 5.0, 2.5, 0.1),
        ParameterSpec("alpha_min_volume_z", 0.5, 2.0, 1.0, 0.1),
        ParameterSpec("alpha_bb_width_threshold", 0.001, 0.01, 0.005, 0.001),
        ParameterSpec("alpha_ema_fast", 8.0, 20.0, 12.0, 1.0, "discrete"),
        ParameterSpec("alpha_ema_slow", 20.0, 55.0, 26.0, 1.0, "discrete"),
    ])

    # OMEGA H1/H4 Swing parameters
    space.register_module("OMEGA", [
        ParameterSpec("omega_trend_ema_period", 50.0, 200.0, 100.0, 10.0, "discrete"),
        ParameterSpec("omega_entry_pullback_pct", 0.001, 0.01, 0.003, 0.001),
        ParameterSpec("omega_atr_sl_mult", 2.0, 5.0, 3.0, 0.5),
        ParameterSpec("omega_atr_tp_mult", 3.0, 8.0, 5.0, 0.5),
        ParameterSpec("omega_trail_atr_mult", 1.5, 4.0, 2.5, 0.5),
        ParameterSpec("omega_min_adx", 15.0, 35.0, 25.0, 1.0),
    ])

    # SENTINEL risk parameters (narrow ranges — these are safety-critical)
    space.register_module("SENTINEL", [
        ParameterSpec("sentinel_size_base_pct", 0.005, 0.02, 0.01, 0.001),
        ParameterSpec("sentinel_max_correlated_pos", 1.0, 3.0, 2.0, 1.0, "discrete"),
    ])

    # ARES consensus parameters
    space.register_module("ARES", [
        ParameterSpec("ares_min_consensus", 0.10, 0.40, 0.20, 0.05),
        ParameterSpec("ares_min_agreement", 0.40, 0.70, 0.50, 0.05),
        ParameterSpec("ares_min_confidence", 0.45, 0.70, 0.55, 0.05),
        ParameterSpec("ares_cooldown_bars", 1.0, 10.0, 3.0, 1.0, "discrete"),
    ])

    return space
