"""
APHELION RL Exit Agent — Reinforcement-Learning-based optimal exit timing.

Uses PPO (Proximal Policy Optimisation) to learn when to close positions,
framing exit timing as an optimal stopping problem.

Exports:
    ExitEnvConfig, ExitEnvironment  — Gym-like environment
    ExitAgentConfig, RLExitAgent    — PPO agent with value baseline
    ExitDecision                    — Agent output dataclass
"""

from aphelion.intelligence.rl_exit.environment import (
    ExitEnvConfig,
    ExitEnvironment,
)
from aphelion.intelligence.rl_exit.agent import (
    ExitAgentConfig,
    RLExitAgent,
    ExitDecision,
)

__all__ = [
    "ExitEnvConfig",
    "ExitEnvironment",
    "ExitAgentConfig",
    "RLExitAgent",
    "ExitDecision",
]
