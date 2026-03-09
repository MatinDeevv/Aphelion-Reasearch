"""
APHELION Money Makers — Position & Money Management Core (Phase 9)

Provides the capital allocation, position sizing, and portfolio-level
money management layer that sits between strategy signals and execution.

Components:
  position_manager  — Dynamic position sizing (Kelly, volatility-targeted, fixed-fractional)
  capital_allocator — Multi-strategy capital allocation with drawdown budgets
  risk_budget       — Portfolio-level risk budgeting per strategy / regime
"""

from aphelion.money.position_manager import (
    SizingMethod,
    PositionManagerConfig,
    PositionManager,
)
from aphelion.money.capital_allocator import (
    AllocationMethod,
    StrategySlot,
    CapitalAllocatorConfig,
    CapitalAllocator,
)
from aphelion.money.risk_budget import (
    RiskBudget,
    RiskBudgetConfig,
    StrategyRiskState,
)

__all__ = [
    "SizingMethod", "PositionManagerConfig", "PositionManager",
    "AllocationMethod", "StrategySlot", "CapitalAllocatorConfig", "CapitalAllocator",
    "RiskBudget", "RiskBudgetConfig", "StrategyRiskState",
]
