"""
Phase 9 Tests — Money Makers: Position Sizing, Capital Allocation, Risk Budget

Covers:
  - PositionManager: all 5 sizing methods, SENTINEL clamp, trade recording
  - CapitalAllocator: all 4 allocation methods, rebalance, DD deactivation
  - RiskBudget: can_trade checks, event handlers, new_day reset
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aphelion.core.config import SENTINEL, KELLY_FRACTION
from aphelion.money.position_manager import (
    SizingMethod, PositionManagerConfig, PositionManager, SizeResult,
)
from aphelion.money.capital_allocator import (
    AllocationMethod, StrategySlot, CapitalAllocatorConfig, CapitalAllocator,
)
from aphelion.money.risk_budget import (
    RiskBudget, RiskBudgetConfig, StrategyRiskState,
)


# ═══════════════════════════════════════════════════════════════════════════
# Position Manager
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionManager:
    def _make_pm(self, method: SizingMethod, **kw) -> PositionManager:
        cfg = PositionManagerConfig(method=method, **kw)
        return PositionManager(cfg)

    # ── Fixed Fractional ────────────────────────────────────────────────
    def test_fixed_fractional_basic(self):
        pm = self._make_pm(SizingMethod.FIXED_FRACTIONAL, fixed_risk_pct=0.015)
        r = pm.compute_size(
            equity=10_000, signal_confidence=1.0,
            atr=5.0, entry_price=2800, sl_distance=10.0,
        )
        assert r.method_used == SizingMethod.FIXED_FRACTIONAL
        assert r.size_pct <= SENTINEL.max_position_pct
        assert r.size_pct > 0
        assert r.size_lots > 0

    def test_fixed_fractional_scales_with_confidence(self):
        pm = self._make_pm(SizingMethod.FIXED_FRACTIONAL, fixed_risk_pct=0.015)
        r_full = pm.compute_size(10_000, 1.0, 5.0, 2800, 10.0)
        r_half = pm.compute_size(10_000, 0.5, 5.0, 2800, 10.0)
        assert r_full.size_pct > r_half.size_pct

    # ── Kelly Criterion ────────────────────────────────────────────────
    def test_kelly_sizing_with_no_history(self):
        pm = self._make_pm(SizingMethod.KELLY)
        r = pm.compute_size(10_000, 0.65, 5.0, 2800, 10.0)
        assert r.size_pct > 0
        assert r.size_pct <= SENTINEL.max_position_pct

    def test_kelly_sizing_adapts_to_history(self):
        pm = self._make_pm(SizingMethod.KELLY, kelly_lookback=20)
        for _ in range(15):
            pm.record_trade(0.005)   # Winning trades
        for _ in range(5):
            pm.record_trade(-0.003)  # Some losses
        r = pm.compute_size(10_000, 0.7, 5.0, 2800, 10.0)
        assert r.size_pct > 0

    # ── Volatility Target ──────────────────────────────────────────────
    def test_vol_target_with_returns(self):
        pm = self._make_pm(SizingMethod.VOLATILITY_TARGET, target_annual_vol=0.15)
        daily_rets = list(np.random.default_rng(42).normal(0, 0.01, 30))
        r = pm.compute_size(10_000, 0.7, 5.0, 2800, 10.0, daily_returns=daily_rets)
        assert r.size_pct > 0

    def test_vol_target_without_data_falls_back(self):
        pm = self._make_pm(SizingMethod.VOLATILITY_TARGET)
        r = pm.compute_size(10_000, 0.7, 5.0, 2800, 10.0, daily_returns=[])
        # Falls back to fixed_risk_pct
        assert r.size_pct > 0

    # ── Anti-Martingale ────────────────────────────────────────────────
    def test_anti_mart_increases_after_win(self):
        pm = self._make_pm(SizingMethod.ANTI_MARTINGALE, fixed_risk_pct=0.01)
        r0 = pm.compute_size(10_000, 0.8, 5.0, 2800, 10.0)
        pm.record_trade(0.01)
        pm.record_trade(0.01)
        r1 = pm.compute_size(10_000, 0.8, 5.0, 2800, 10.0)
        assert r1.raw_fraction >= r0.raw_fraction

    def test_anti_mart_decreases_after_loss(self):
        pm = self._make_pm(SizingMethod.ANTI_MARTINGALE, fixed_risk_pct=0.015)
        pm.record_trade(-0.005)
        pm.record_trade(-0.005)
        r = pm.compute_size(10_000, 0.8, 5.0, 2800, 10.0)
        assert r.raw_fraction <= 0.015

    # ── Optimal-f ──────────────────────────────────────────────────────
    def test_optimal_f_with_history(self):
        pm = self._make_pm(SizingMethod.OPTIMAL_F, optf_lookback=50)
        rng = np.random.default_rng(42)
        for _ in range(40):
            pm.record_trade(float(rng.normal(0.002, 0.005)))
        r = pm.compute_size(10_000, 0.7, 5.0, 2800, 10.0)
        assert r.size_pct > 0
        assert r.size_pct <= SENTINEL.max_position_pct

    def test_optimal_f_without_data_falls_back(self):
        pm = self._make_pm(SizingMethod.OPTIMAL_F)
        r = pm.compute_size(10_000, 0.7, 5.0, 2800, 10.0)
        assert r.size_pct > 0

    # ── SENTINEL Clamp ─────────────────────────────────────────────────
    def test_sentinel_clamp(self):
        pm = self._make_pm(SizingMethod.FIXED_FRACTIONAL, fixed_risk_pct=0.99)
        r = pm.compute_size(10_000, 1.0, 5.0, 2800, 10.0)
        assert r.size_pct <= SENTINEL.max_position_pct

    # ── Reset ──────────────────────────────────────────────────────────
    def test_reset_clears_state(self):
        pm = self._make_pm(SizingMethod.KELLY)
        pm.record_trade(0.01)
        pm.record_trade(-0.005)
        pm.reset()
        assert pm._recent_trades == []
        assert pm._wins_streak == 0


# ═══════════════════════════════════════════════════════════════════════════
# Capital Allocator
# ═══════════════════════════════════════════════════════════════════════════

class TestCapitalAllocator:
    def _make_allocator(self, method: AllocationMethod, equity: float = 100_000) -> CapitalAllocator:
        cfg = CapitalAllocatorConfig(method=method, rebalance_interval_bars=10)
        return CapitalAllocator(equity, cfg)

    def _make_slot(self, sid: str, name: str, **kw) -> StrategySlot:
        return StrategySlot(strategy_id=sid, name=name, **kw)

    # ── Equal Weight ───────────────────────────────────────────────────
    def test_equal_weight(self):
        ca = self._make_allocator(AllocationMethod.EQUAL_WEIGHT)
        ca.register_strategy(self._make_slot("A", "Alpha"))
        ca.register_strategy(self._make_slot("B", "Bravo"))
        w = ca.rebalance()
        assert len(w) == 2
        assert w["A"] == pytest.approx(w["B"], abs=0.01)

    # ── Risk Parity ────────────────────────────────────────────────────
    def test_risk_parity(self):
        ca = self._make_allocator(AllocationMethod.RISK_PARITY)
        s1 = self._make_slot("A", "Low Vol", recent_volatility=0.05)
        s2 = self._make_slot("B", "High Vol", recent_volatility=0.20)
        ca.register_strategy(s1)
        ca.register_strategy(s2)
        w = ca.rebalance()
        # Low-vol strategy gets higher allocation
        assert w["A"] > w["B"]

    # ── Performance Weighted ───────────────────────────────────────────
    def test_performance_weighted(self):
        ca = self._make_allocator(AllocationMethod.PERFORMANCE_WEIGHTED)
        s1 = self._make_slot("A", "Winner", recent_sharpe=2.0)
        s2 = self._make_slot("B", "Loser", recent_sharpe=0.1)
        ca.register_strategy(s1)
        ca.register_strategy(s2)
        w = ca.rebalance()
        assert w["A"] > w["B"]

    # ── Dynamic CPPI ───────────────────────────────────────────────────
    def test_cppi_returns_allocations(self):
        ca = self._make_allocator(AllocationMethod.DYNAMIC_CPPI)
        ca.register_strategy(self._make_slot("A", "Alpha"))
        ca.register_strategy(self._make_slot("B", "Bravo"))
        w = ca.rebalance()
        assert len(w) == 2
        assert sum(w.values()) <= 1.0 + 1e-6

    # ── Drawdown Budget ────────────────────────────────────────────────
    def test_drawdown_budget_deactivation(self):
        ca = self._make_allocator(AllocationMethod.EQUAL_WEIGHT)
        slot = self._make_slot("A", "Alpha", max_drawdown_budget=0.05)
        slot.peak_capital = 10_000
        slot.allocated_capital = 9_000  # 10% DD > 5% budget
        slot.current_drawdown = 0.10
        ca.register_strategy(slot)
        ca.on_bar()  # Triggers DD check
        assert not ca.slots["A"].is_active

    # ── Rebalance Timing ───────────────────────────────────────────────
    def test_rebalance_happens_on_interval(self):
        ca = self._make_allocator(AllocationMethod.EQUAL_WEIGHT)
        ca.register_strategy(self._make_slot("A", "Alpha"))
        for _ in range(9):
            assert not ca.on_bar()
        assert ca.on_bar()  # 10th bar triggers rebalance

    # ── Min/Max Constraints ────────────────────────────────────────────
    def test_allocation_constraints(self):
        cfg = CapitalAllocatorConfig(
            method=AllocationMethod.EQUAL_WEIGHT,
            min_allocation_pct=0.05,
            max_allocation_pct=0.40,
        )
        ca = CapitalAllocator(100_000, cfg)
        ca.register_strategy(self._make_slot("A", "A"))
        w = ca.rebalance()
        assert w["A"] >= 0.05
        assert w["A"] <= 0.40

    # ── Strategy Return Tracking ───────────────────────────────────────
    def test_update_strategy_return(self):
        ca = self._make_allocator(AllocationMethod.EQUAL_WEIGHT)
        slot = self._make_slot("A", "Alpha")
        slot.allocated_capital = 50_000
        slot.peak_capital = 50_000
        ca.register_strategy(slot)
        ca.update_strategy_return("A", -0.02)  # 2% loss
        assert ca.slots["A"].daily_returns[-1] == -0.02
        assert ca.slots["A"].current_drawdown > 0


# ═══════════════════════════════════════════════════════════════════════════
# Risk Budget
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskBudget:
    def _make_rb(self, equity: float = 100_000, **kw) -> RiskBudget:
        cfg = RiskBudgetConfig(**kw)
        return RiskBudget(equity, cfg)

    def test_register_strategy(self):
        rb = self._make_rb()
        state = rb.register_strategy("HYDRA")
        assert state.strategy_id == "HYDRA"
        assert "HYDRA" in rb.strategies

    def test_can_trade_passes_for_registered(self):
        rb = self._make_rb()
        rb.register_strategy("HYDRA")
        allowed, reason = rb.can_trade("HYDRA")
        assert allowed
        assert reason == ""

    def test_can_trade_fails_for_unregistered(self):
        rb = self._make_rb()
        allowed, reason = rb.can_trade("UNKNOWN")
        assert not allowed
        assert "NOT_REGISTERED" in reason

    def test_daily_loss_limit_halts_strategy(self):
        rb = self._make_rb(equity=100_000, per_strategy_daily_loss_pct=0.01)
        rb.register_strategy("HYDRA")
        # Record losses exceeding the 1% limit (100_000 * 0.01 = 1000)
        rb.on_trade_close("HYDRA", -1100.0)
        allowed, reason = rb.can_trade("HYDRA")
        assert not allowed
        assert "DAILY_LOSS" in reason

    def test_max_trades_per_day(self):
        rb = self._make_rb(per_strategy_max_trades=3)
        rb.register_strategy("HYDRA")
        for _ in range(3):
            rb.on_trade_open("HYDRA", 100.0)
        allowed, reason = rb.can_trade("HYDRA")
        assert not allowed
        assert "MAX_TRADES" in reason

    def test_global_max_positions(self):
        rb = self._make_rb(global_max_open_positions=2)
        rb.register_strategy("A")
        rb.register_strategy("B")
        rb.on_trade_open("A", 100)
        rb.on_trade_open("B", 100)
        allowed, reason = rb.can_trade("A")
        assert not allowed
        assert "GLOBAL_MAX" in reason

    def test_portfolio_daily_loss_halts(self):
        rb = self._make_rb(
            equity=100_000,
            max_portfolio_daily_loss_pct=0.02,
            per_strategy_daily_loss_pct=0.10,  # High per-strategy limit so portfolio limit triggers first
        )
        rb.register_strategy("HYDRA")
        rb.on_trade_close("HYDRA", -2100.0)  # exceeds 2% portfolio daily limit
        allowed, reason = rb.can_trade("HYDRA")
        assert not allowed
        assert "PORTFOLIO" in reason or "DAILY_LOSS" in reason

    def test_new_day_resets(self):
        from datetime import datetime, timezone
        rb = self._make_rb(per_strategy_max_trades=2)
        rb.register_strategy("HYDRA")
        rb.on_trade_open("HYDRA", 100)
        rb.on_trade_open("HYDRA", 100)
        allowed, _ = rb.can_trade("HYDRA")
        assert not allowed
        # New day resets
        rb.on_new_day(datetime(2024, 1, 16, tzinfo=timezone.utc))
        allowed2, _ = rb.can_trade("HYDRA")
        assert allowed2

    def test_update_equity(self):
        rb = self._make_rb(equity=100_000)
        rb.register_strategy("HYDRA")
        rb.update_equity(105_000)
        assert rb._equity == 105_000
        assert rb._peak_equity == 105_000

    def test_risk_summary(self):
        rb = self._make_rb()
        rb.register_strategy("HYDRA")
        summary = rb.get_risk_summary()
        assert "equity" in summary
        assert "strategies" in summary
        assert "HYDRA" in summary["strategies"]

    def test_portfolio_drawdown_halts(self):
        rb = self._make_rb(equity=100_000, max_portfolio_drawdown_pct=0.05)
        rb.register_strategy("HYDRA")
        rb._peak_equity = 100_000
        rb._equity = 94_000  # 6% drawdown > 5% limit
        allowed, reason = rb.can_trade("HYDRA")
        assert not allowed
        assert "DRAWDOWN" in reason

    def test_reset(self):
        rb = self._make_rb()
        rb.register_strategy("HYDRA")
        rb.on_trade_open("HYDRA", 100)
        rb._portfolio_halted = True
        rb.reset()
        assert not rb.is_halted
        assert rb._global_open_positions == 0
