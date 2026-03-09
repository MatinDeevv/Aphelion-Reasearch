"""
APHELION Money Makers — Multi-Strategy Capital Allocator

Distributes total account equity across multiple strategy slots
using performance-weighted allocation, drawdown budgets, and
automatic rebalancing.

Allocation methods:
  - Equal Weight: each strategy gets 1/N of capital
  - Risk Parity: allocate inversely proportional to volatility
  - Performance Weighted: allocate proportional to recent Sharpe
  - Dynamic CPPI: Constant Proportion Portfolio Insurance
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from aphelion.core.config import SENTINEL

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    EQUAL_WEIGHT = auto()
    RISK_PARITY = auto()
    PERFORMANCE_WEIGHTED = auto()
    DYNAMIC_CPPI = auto()


@dataclass
class StrategySlot:
    """Represents a strategy with its allocation and performance state."""
    strategy_id: str
    name: str
    allocated_pct: float = 0.0              # Fraction of total equity
    allocated_capital: float = 0.0          # Dollar amount
    current_drawdown: float = 0.0           # Current drawdown from peak
    peak_capital: float = 0.0
    recent_sharpe: float = 0.0
    recent_volatility: float = 0.15         # Annualised
    trade_count: int = 0
    win_rate: float = 0.5
    is_active: bool = True
    daily_returns: list[float] = field(default_factory=list)
    max_drawdown_budget: float = 0.10       # Kill strategy at 10% DD
    # Performance history
    cumulative_pnl: float = 0.0


@dataclass
class CapitalAllocatorConfig:
    """Configuration for the capital allocator."""
    method: AllocationMethod = AllocationMethod.PERFORMANCE_WEIGHTED
    rebalance_interval_bars: int = 500      # Rebalance every N bars
    min_allocation_pct: float = 0.05        # Minimum 5% per strategy
    max_allocation_pct: float = 0.40        # Maximum 40% per strategy
    # Performance weighting
    sharpe_lookback_days: int = 30
    min_sharpe_for_allocation: float = -0.5
    # CPPI
    cppi_multiplier: float = 3.0
    cppi_floor_pct: float = 0.80            # Protect 80% of equity
    # Drawdown budget enforcement
    enforce_dd_budget: bool = True


class CapitalAllocator:
    """
    Distributes account capital across multiple strategy slots.
    Supports performance-weighted rebalancing with drawdown budgets.
    """

    def __init__(
        self,
        total_equity: float,
        config: Optional[CapitalAllocatorConfig] = None,
    ):
        self._config = config or CapitalAllocatorConfig()
        self._total_equity = total_equity
        self._slots: dict[str, StrategySlot] = {}
        self._bars_since_rebalance: int = 0
        self._rebalance_count: int = 0

    # ── Strategy Registration ────────────────────────────────────────────────

    def register_strategy(self, slot: StrategySlot) -> None:
        """Register a new strategy slot."""
        self._slots[slot.strategy_id] = slot
        logger.info("Registered strategy slot: %s (%s)", slot.strategy_id, slot.name)

    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from allocation."""
        self._slots.pop(strategy_id, None)

    # ── Equity Update ────────────────────────────────────────────────────────

    def update_equity(self, total_equity: float) -> None:
        """Update total account equity."""
        self._total_equity = total_equity

    def update_strategy_return(self, strategy_id: str, daily_return: float) -> None:
        """Record a daily return for a strategy slot."""
        slot = self._slots.get(strategy_id)
        if slot is None:
            return
        slot.daily_returns.append(daily_return)
        # Keep bounded
        if len(slot.daily_returns) > 500:
            slot.daily_returns = slot.daily_returns[-500:]
        # Update running metrics
        slot.cumulative_pnl += daily_return * slot.allocated_capital
        slot.allocated_capital *= (1 + daily_return)
        if slot.allocated_capital > slot.peak_capital:
            slot.peak_capital = slot.allocated_capital
        if slot.peak_capital > 0:
            slot.current_drawdown = 1 - slot.allocated_capital / slot.peak_capital

    # ── Rebalance ────────────────────────────────────────────────────────────

    def on_bar(self) -> bool:
        """Call each bar. Returns True if a rebalance occurred."""
        self._bars_since_rebalance += 1

        # Drawdown budget enforcement (always active)
        if self._config.enforce_dd_budget:
            self._enforce_drawdown_budgets()

        if self._bars_since_rebalance >= self._config.rebalance_interval_bars:
            self.rebalance()
            self._bars_since_rebalance = 0
            return True
        return False

    def rebalance(self) -> dict[str, float]:
        """
        Rebalance capital across active strategies.
        Returns dict of strategy_id → allocated_pct.
        """
        active = {k: v for k, v in self._slots.items() if v.is_active}
        if not active:
            return {}

        method = self._config.method

        if method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(active)
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity(active)
        elif method == AllocationMethod.PERFORMANCE_WEIGHTED:
            weights = self._performance_weighted(active)
        elif method == AllocationMethod.DYNAMIC_CPPI:
            weights = self._dynamic_cppi(active)
        else:
            weights = self._equal_weight(active)

        # Apply min/max constraints
        weights = self._apply_constraints(weights)

        # Update slots
        for sid, pct in weights.items():
            self._slots[sid].allocated_pct = pct
            self._slots[sid].allocated_capital = self._total_equity * pct
            if self._slots[sid].peak_capital < self._slots[sid].allocated_capital:
                self._slots[sid].peak_capital = self._slots[sid].allocated_capital

        self._rebalance_count += 1
        logger.info(
            "Rebalance #%d | %s | %s",
            self._rebalance_count, method.name,
            {k: f"{v:.1%}" for k, v in weights.items()},
        )
        return weights

    # ── Allocation Methods ───────────────────────────────────────────────────

    def _equal_weight(self, active: dict[str, StrategySlot]) -> dict[str, float]:
        n = len(active)
        w = 1.0 / n if n > 0 else 0.0
        return {sid: w for sid in active}

    def _risk_parity(self, active: dict[str, StrategySlot]) -> dict[str, float]:
        """Allocate inversely proportional to recent volatility."""
        vols = {}
        for sid, slot in active.items():
            if len(slot.daily_returns) >= 5:
                vol = float(np.std(slot.daily_returns[-30:], ddof=1)) * math.sqrt(252)
            else:
                vol = slot.recent_volatility
            vols[sid] = max(vol, 1e-6)

        inv_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total = sum(inv_vols.values())
        if total <= 0:
            return self._equal_weight(active)
        return {sid: iv / total for sid, iv in inv_vols.items()}

    def _performance_weighted(self, active: dict[str, StrategySlot]) -> dict[str, float]:
        """Allocate proportional to recent Sharpe ratio (shifted to positive)."""
        sharpes = {}
        for sid, slot in active.items():
            if len(slot.daily_returns) >= 10:
                rets = slot.daily_returns[-self._config.sharpe_lookback_days:]
                mean_r = np.mean(rets)
                std_r = np.std(rets, ddof=1)
                sharpe = float(mean_r / std_r * math.sqrt(252)) if std_r > 1e-10 else 0.0
            else:
                sharpe = slot.recent_sharpe
            sharpes[sid] = sharpe

        # Shift so all positive (min sharpe → small positive weight)
        min_sharpe = min(sharpes.values())
        shifted = {sid: s - min_sharpe + 0.1 for sid, s in sharpes.items()}
        total = sum(shifted.values())
        if total <= 0:
            return self._equal_weight(active)
        return {sid: v / total for sid, v in shifted.items()}

    def _dynamic_cppi(self, active: dict[str, StrategySlot]) -> dict[str, float]:
        """
        Constant Proportion Portfolio Insurance.
        Risky allocation = multiplier * (equity - floor).
        Distributes the risky budget equally across strategies.
        """
        floor_value = self._total_equity * self._config.cppi_floor_pct
        cushion = max(self._total_equity - floor_value, 0)
        risky_budget = min(cushion * self._config.cppi_multiplier, self._total_equity)
        risky_pct = risky_budget / self._total_equity if self._total_equity > 0 else 0.0

        n = len(active)
        per_strategy = risky_pct / n if n > 0 else 0.0
        return {sid: per_strategy for sid in active}

    # ── Constraints ──────────────────────────────────────────────────────────

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Enforce min/max allocation per strategy and re-normalise."""
        constrained = {}
        for sid, w in weights.items():
            cw = np.clip(w, self._config.min_allocation_pct, self._config.max_allocation_pct)
            constrained[sid] = float(cw)

        total = sum(constrained.values())
        if total > 1.0:
            constrained = {sid: w / total for sid, w in constrained.items()}
        return constrained

    def _enforce_drawdown_budgets(self) -> None:
        """Deactivate strategies that exceed their drawdown budget."""
        for sid, slot in self._slots.items():
            if slot.is_active and slot.current_drawdown > slot.max_drawdown_budget:
                slot.is_active = False
                logger.warning(
                    "Strategy %s (%s) deactivated: drawdown %.2f%% exceeds budget %.2f%%",
                    sid, slot.name,
                    slot.current_drawdown * 100, slot.max_drawdown_budget * 100,
                )

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def slots(self) -> dict[str, StrategySlot]:
        return dict(self._slots)

    @property
    def active_slots(self) -> dict[str, StrategySlot]:
        return {k: v for k, v in self._slots.items() if v.is_active}

    def get_allocation(self, strategy_id: str) -> float:
        """Get current allocation % for a strategy."""
        slot = self._slots.get(strategy_id)
        return slot.allocated_pct if slot else 0.0

    def get_capital(self, strategy_id: str) -> float:
        """Get current capital allocation for a strategy."""
        slot = self._slots.get(strategy_id)
        return slot.allocated_capital if slot else 0.0

    @property
    def total_allocated_pct(self) -> float:
        return sum(s.allocated_pct for s in self._slots.values() if s.is_active)
