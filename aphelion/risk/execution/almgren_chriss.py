"""
APHELION Almgren-Chriss Optimal Execution Model

Solves WHEN and HOW to execute to minimise market impact + timing risk.
Mathematically optimal execution scheduling for large orders.

The Problem:
  You want to buy/sell X lots over T intervals.
  - Trading too fast → high market impact (you move the price against yourself)
  - Trading too slow → high timing risk (price drifts away during execution)
  Almgren-Chriss finds the optimal trajectory that minimises:
      E[cost] + λ * Var[cost]

Model Parameters:
  - σ (volatility): price uncertainty per unit time
  - η (temporary impact): linear coefficient of instantaneous price impact
  - γ (permanent impact): linear coefficient of permanent price impact
  - λ (risk aversion): tradeoff between expected cost and variance

Output:
  - Optimal trade schedule: how many lots to trade at each interval
  - Expected execution cost (implementation shortfall)
  - Cost variance (execution risk)
  - VWAP comparison: expected performance vs naive VWAP

References:
  - Almgren & Chriss (2000) "Optimal Execution of Portfolio Transactions"
  - Almgren (2003) "Optimal Execution with Nonlinear Impact Functions"
  - Gatheral (2010) "No-Dynamic-Arbitrage and Market Impact"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ExecutionConfig:
    """Configuration for the Almgren-Chriss execution model."""

    # Market microstructure parameters
    volatility: float = 0.0          # σ: annualised volatility (will be estimated if 0)
    temporary_impact: float = 0.0    # η: temporary impact coefficient
    permanent_impact: float = 0.0    # γ: permanent impact coefficient

    # Risk aversion
    risk_aversion: float = 1e-6      # λ: higher = more aggressive (front-loaded)

    # Execution parameters
    n_intervals: int = 20            # Number of time slices for execution
    interval_seconds: float = 60.0   # Duration of each interval (seconds)

    # Auto-estimation
    auto_estimate_impact: bool = True  # Estimate η, γ from spread and volume


@dataclass
class ExecutionPlan:
    """Output of the Almgren-Chriss optimal execution solver."""

    # Schedule: how many lots to trade at each interval
    trade_schedule: list[float] = field(default_factory=list)
    # Remaining position at each interval boundary
    position_trajectory: list[float] = field(default_factory=list)

    # Cost analysis
    expected_cost: float = 0.0           # E[implementation shortfall] in price units
    cost_variance: float = 0.0           # Var[implementation shortfall]
    cost_std: float = 0.0                # Std dev of cost

    # Cost breakdown
    permanent_impact_cost: float = 0.0   # Cost from permanent market impact
    temporary_impact_cost: float = 0.0   # Cost from temporary impact
    timing_risk_cost: float = 0.0        # Cost from price drift risk

    # Comparison benchmarks
    naive_vwap_cost: float = 0.0         # Cost of naive uniform VWAP
    improvement_vs_vwap_pct: float = 0.0 # How much better than VWAP (%)

    # Parameters used
    risk_aversion: float = 0.0
    total_lots: float = 0.0
    n_intervals: int = 0
    urgency: float = 0.0                # κ*T: dimensionless urgency parameter

    @property
    def is_front_loaded(self) -> bool:
        """True if the schedule trades more in early intervals (high urgency)."""
        if not self.trade_schedule or len(self.trade_schedule) < 2:
            return False
        mid = len(self.trade_schedule) // 2
        first_half = sum(abs(t) for t in self.trade_schedule[:mid])
        second_half = sum(abs(t) for t in self.trade_schedule[mid:])
        return first_half > second_half * 1.1


@dataclass
class ImpactEstimate:
    """Market impact parameters estimated from observable data."""
    temporary_impact: float = 0.0     # η
    permanent_impact: float = 0.0     # γ
    volatility_per_interval: float = 0.0
    bid_ask_spread: float = 0.0
    avg_volume: float = 0.0


# ─── Impact Estimator ────────────────────────────────────────────────────────


class MarketImpactEstimator:
    """
    Estimates market impact parameters from observable market data.

    Uses the square-root impact model calibrated to spread and volume:
      η ≈ spread / (2 * σ * √(V_avg))   (temporary)
      γ ≈ η * participation_rate          (permanent, fraction of temporary)

    This is a simplified but practical approach used by many execution desks.
    """

    @staticmethod
    def estimate(
        bid_ask_spread: float,
        avg_volume_per_interval: float,
        volatility_per_interval: float,
        order_size: float,
    ) -> ImpactEstimate:
        """
        Estimate impact parameters from market observables.

        Args:
            bid_ask_spread: Average bid-ask spread in price units
            avg_volume_per_interval: Average volume per execution interval
            volatility_per_interval: Price volatility per interval (σ√τ)
            order_size: Total order size in lots
        """
        if avg_volume_per_interval <= 0 or volatility_per_interval <= 0:
            return ImpactEstimate(
                temporary_impact=1e-6,
                permanent_impact=1e-7,
                volatility_per_interval=volatility_per_interval,
                bid_ask_spread=bid_ask_spread,
                avg_volume=avg_volume_per_interval,
            )

        # Temporary impact: half-spread normalised by volatility and volume
        eta = bid_ask_spread / (2.0 * volatility_per_interval * np.sqrt(avg_volume_per_interval))
        eta = max(eta, 1e-8)

        # Permanent impact: participation rate * temporary
        participation = order_size / (avg_volume_per_interval * 20)  # Over ~20 intervals
        gamma = eta * min(participation, 0.5) * 0.1  # Permanent is fraction of temporary
        gamma = max(gamma, 1e-9)

        return ImpactEstimate(
            temporary_impact=eta,
            permanent_impact=gamma,
            volatility_per_interval=volatility_per_interval,
            bid_ask_spread=bid_ask_spread,
            avg_volume=avg_volume_per_interval,
        )


# ─── Almgren-Chriss Solver ───────────────────────────────────────────────────


class AlmgrenChrissSolver:
    """
    Optimal execution trajectory solver.

    Minimises:  E[cost] + λ * Var[cost]

    where cost is the implementation shortfall between the decision price
    and the actual average execution price.

    The closed-form solution is:

        x_j = X * sinh(κ(T-t_j)) / sinh(κT)

    where:
        κ = sqrt(λσ² / η)  (urgency parameter)
        X = total shares to execute
        T = total execution time
        t_j = time at interval j

    Usage::

        solver = AlmgrenChrissSolver(ExecutionConfig(
            risk_aversion=1e-6,
            n_intervals=20,
        ))

        # With known impact parameters
        plan = solver.solve(
            total_lots=10.0,
            volatility=0.01,
            temporary_impact=1e-5,
            permanent_impact=1e-6,
        )

        # With auto-estimation from market data
        plan = solver.solve_from_market(
            total_lots=10.0,
            bid_ask_spread=0.50,
            avg_volume=1000.0,
            volatility=0.01,
        )

        print(plan.trade_schedule)
        print(f"Expected cost: {plan.expected_cost:.2f}")
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self._cfg = config or ExecutionConfig()

    def solve(
        self,
        total_lots: float,
        volatility: float,
        temporary_impact: float,
        permanent_impact: float,
        risk_aversion: Optional[float] = None,
        n_intervals: Optional[int] = None,
    ) -> ExecutionPlan:
        """
        Solve for the optimal execution trajectory.

        Args:
            total_lots: Total position to execute (positive = buy, negative = sell)
            volatility: Price volatility per interval (σ√τ)
            temporary_impact: η coefficient
            permanent_impact: γ coefficient
            risk_aversion: Override config risk aversion
            n_intervals: Override config number of intervals
        """
        lam = risk_aversion if risk_aversion is not None else self._cfg.risk_aversion
        N = n_intervals if n_intervals is not None else self._cfg.n_intervals
        X = abs(total_lots)
        sign = 1.0 if total_lots >= 0 else -1.0

        if X < 1e-12 or N < 1:
            return ExecutionPlan(total_lots=total_lots, n_intervals=N)

        eta = max(temporary_impact, 1e-12)
        gamma = max(permanent_impact, 0.0)
        sigma = max(volatility, 1e-12)
        tau = 1.0  # Normalised interval length

        # Urgency parameter
        kappa_sq = lam * sigma ** 2 / eta
        kappa = np.sqrt(max(kappa_sq, 1e-20))
        kappa_T = kappa * N * tau

        # Optimal position trajectory: x_j = X * sinh(κ(T - t_j)) / sinh(κT)
        # where t_j = j * tau, T = N * tau
        T = N * tau
        trajectory = np.zeros(N + 1)
        trajectory[0] = X

        sinh_kT = np.sinh(kappa_T)
        if abs(sinh_kT) < 1e-20:
            # κ ≈ 0 → risk-neutral → uniform VWAP
            trajectory = np.linspace(X, 0, N + 1)
        else:
            for j in range(N + 1):
                t_j = j * tau
                trajectory[j] = X * np.sinh(kappa * (T - t_j)) / sinh_kT

        # Trade schedule: n_j = x_{j-1} - x_j  (lots traded in interval j)
        schedule = np.diff(-trajectory)  # negative diff gives lots sold

        # ── Cost analysis ─────────────────────────────────────────────────

        # Permanent impact cost: γ * X^2 / 2
        perm_cost = gamma * X ** 2 / 2.0

        # Temporary impact cost: η * Σ(n_j^2)
        temp_cost = eta * float(np.sum(schedule ** 2))

        # Timing risk (variance of cost): σ^2 * Σ(x_j^2 * tau)
        timing_var = sigma ** 2 * tau * float(np.sum(trajectory[:-1] ** 2))

        expected_cost = perm_cost + temp_cost
        total_cost = expected_cost + lam * timing_var

        # Naive VWAP benchmark: uniform n_j = X / N
        vwap_schedule = np.ones(N) * X / N
        vwap_temp_cost = eta * float(np.sum(vwap_schedule ** 2))
        vwap_traj = np.linspace(X, 0, N + 1)
        vwap_timing = sigma ** 2 * tau * float(np.sum(vwap_traj[:-1] ** 2))
        naive_vwap_cost = perm_cost + vwap_temp_cost + lam * vwap_timing

        improvement = 0.0
        if naive_vwap_cost > 0:
            improvement = (naive_vwap_cost - total_cost) / naive_vwap_cost * 100

        return ExecutionPlan(
            trade_schedule=(schedule * sign).tolist(),
            position_trajectory=(trajectory * sign).tolist(),
            expected_cost=expected_cost,
            cost_variance=timing_var,
            cost_std=np.sqrt(timing_var),
            permanent_impact_cost=perm_cost,
            temporary_impact_cost=temp_cost,
            timing_risk_cost=lam * timing_var,
            naive_vwap_cost=naive_vwap_cost,
            improvement_vs_vwap_pct=improvement,
            risk_aversion=lam,
            total_lots=total_lots,
            n_intervals=N,
            urgency=float(kappa_T),
        )

    def solve_from_market(
        self,
        total_lots: float,
        bid_ask_spread: float,
        avg_volume: float,
        volatility: float,
        risk_aversion: Optional[float] = None,
    ) -> ExecutionPlan:
        """
        Solve using auto-estimated impact parameters from market observables.

        Args:
            total_lots: Total position to execute
            bid_ask_spread: Current bid-ask spread
            avg_volume: Average volume per interval
            volatility: Volatility per interval
        """
        impact = MarketImpactEstimator.estimate(
            bid_ask_spread=bid_ask_spread,
            avg_volume_per_interval=avg_volume,
            volatility_per_interval=volatility,
            order_size=abs(total_lots),
        )
        return self.solve(
            total_lots=total_lots,
            volatility=volatility,
            temporary_impact=impact.temporary_impact,
            permanent_impact=impact.permanent_impact,
            risk_aversion=risk_aversion,
        )

    def adaptive_urgency(
        self,
        remaining_lots: float,
        remaining_intervals: int,
        current_volatility: float,
        temporary_impact: float,
        risk_aversion: Optional[float] = None,
    ) -> float:
        """
        Compute the optimal trade size for the NEXT interval given
        remaining position and time. Used for real-time re-optimisation.

        Returns the number of lots to trade in the next interval.
        """
        if remaining_intervals <= 0 or abs(remaining_lots) < 1e-12:
            return remaining_lots  # Trade everything remaining

        lam = risk_aversion if risk_aversion is not None else self._cfg.risk_aversion
        eta = max(temporary_impact, 1e-12)
        sigma = max(current_volatility, 1e-12)

        kappa = np.sqrt(lam * sigma ** 2 / eta)
        kappa_T = kappa * remaining_intervals

        sinh_kT = np.sinh(kappa_T)
        if abs(sinh_kT) < 1e-20:
            return remaining_lots / remaining_intervals

        # Optimal next trade: n_1 = X * (cosh(κT) - cosh(κ(T-1))) / sinh(κT)
        # Simplified: n_1 = X * sinh(κ) / sinh(κT) * cosh(κ(T-1))
        # Most practical form:
        x_0 = abs(remaining_lots)
        x_1 = x_0 * np.sinh(kappa * (remaining_intervals - 1)) / sinh_kT
        trade = (x_0 - x_1) * np.sign(remaining_lots)
        return float(trade)


# ─── Execution Monitor ───────────────────────────────────────────────────────


class ExecutionMonitor:
    """
    Tracks actual execution vs planned trajectory and provides
    real-time slippage analysis and re-optimisation triggers.
    """

    def __init__(self, plan: ExecutionPlan):
        self._plan = plan
        self._actual_trades: list[float] = []
        self._actual_prices: list[float] = []
        self._decision_price: float = 0.0
        self._interval: int = 0

    def set_decision_price(self, price: float) -> None:
        """Set the arrival/decision price for IS calculation."""
        self._decision_price = price

    def record_fill(self, lots: float, avg_price: float) -> dict:
        """
        Record an actual fill and return execution quality metrics.
        """
        self._actual_trades.append(lots)
        self._actual_prices.append(avg_price)
        self._interval += 1

        planned = self._plan.trade_schedule[self._interval - 1] if self._interval <= len(self._plan.trade_schedule) else 0.0
        slippage = lots - planned

        # Implementation shortfall so far
        if self._decision_price > 0 and self._actual_trades:
            total_traded = sum(self._actual_trades)
            if total_traded != 0:
                wavg_price = sum(t * p for t, p in zip(self._actual_trades, self._actual_prices)) / total_traded
                is_bps = (wavg_price - self._decision_price) / self._decision_price * 10_000
            else:
                is_bps = 0.0
        else:
            is_bps = 0.0

        return {
            "interval": self._interval,
            "planned_lots": planned,
            "actual_lots": lots,
            "schedule_slippage": slippage,
            "implementation_shortfall_bps": is_bps,
            "remaining_intervals": len(self._plan.trade_schedule) - self._interval,
            "completion_pct": self._interval / max(len(self._plan.trade_schedule), 1) * 100,
        }

    @property
    def total_executed(self) -> float:
        return sum(self._actual_trades)

    @property
    def remaining(self) -> float:
        return self._plan.total_lots - self.total_executed
