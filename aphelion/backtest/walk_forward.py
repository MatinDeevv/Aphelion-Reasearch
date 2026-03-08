"""
APHELION Walk-Forward Validation Engine
Rolling out-of-sample validation: 6-month train, 2-month test, 1-month step.
Minimum 12 OOS windows required for deployment approval.
Prevents overfitting by validating genome performance on unseen data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from aphelion.backtest.engine import BacktestConfig, BacktestEngine
from aphelion.backtest.metrics import BacktestMetrics, compute_metrics
from aphelion.backtest.monte_carlo import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResults,
)
from aphelion.backtest.order import BacktestTrade
from aphelion.core.data_layer import Bar, DataLayer

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_bars: int = 259_200        # ~6 months of M1 bars (6 * 30 * 24 * 60)
    test_bars: int = 86_400          # ~2 months of M1 bars (2 * 30 * 24 * 60)
    step_bars: int = 43_200          # ~1 month step (1 * 30 * 24 * 60)
    min_windows: int = 12            # Minimum OOS windows required
    min_trades_per_window: int = 20  # Minimum trades per OOS window
    monte_carlo_paths: int = 500     # MC paths per window
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class WalkForwardWindow:
    """Results from a single walk-forward window."""
    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_bars_count: int
    test_bars_count: int
    train_metrics: Optional[BacktestMetrics] = None
    test_metrics: Optional[BacktestMetrics] = None
    test_trades: list[BacktestTrade] = field(default_factory=list)
    test_equity_curve: list[float] = field(default_factory=list)
    mc_results: Optional[MonteCarloResults] = None

    @property
    def oos_sharpe(self) -> float:
        return self.test_metrics.sharpe if self.test_metrics else 0.0

    @property
    def oos_return_pct(self) -> float:
        return self.test_metrics.total_return_pct if self.test_metrics else 0.0

    @property
    def oos_mdd_pct(self) -> float:
        return self.test_metrics.max_drawdown_pct if self.test_metrics else 0.0

    @property
    def oos_trade_count(self) -> int:
        return self.test_metrics.total_trades if self.test_metrics else 0


@dataclass
class WalkForwardResults:
    """Aggregate results from full walk-forward validation."""
    config: WalkForwardConfig
    windows: list[WalkForwardWindow] = field(default_factory=list)
    total_oos_trades: int = 0
    combined_oos_metrics: Optional[BacktestMetrics] = None
    combined_mc: Optional[MonteCarloResults] = None

    # Aggregate OOS statistics
    avg_oos_sharpe: float = 0.0
    min_oos_sharpe: float = 0.0
    max_oos_sharpe: float = 0.0
    std_oos_sharpe: float = 0.0

    avg_oos_return_pct: float = 0.0
    avg_oos_mdd_pct: float = 0.0
    oos_win_rate_pct: float = 0.0
    profitable_windows: int = 0
    profitable_window_ratio: float = 0.0
    median_oos_return_pct: float = 0.0
    combined_profitability_score: float = 0.0

    # Deployment decision
    deployment_approved: bool = False
    deployment_reason: str = ""

    @property
    def num_windows(self) -> int:
        return len(self.windows)


class WalkForwardEngine:
    """
    Rolling walk-forward validation engine.
    Splits bar data into overlapping train/test windows,
    runs a backtest on each, collects OOS metrics,
    and produces a deployment approval decision.
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        sentinel_stack: dict,
        data_layer: DataLayer,
        strategy_factory: Callable,
    ):
        """
        Args:
            config: Walk-forward configuration.
            sentinel_stack: SENTINEL components dict (core, validator, cb, enforcer, sizer).
            data_layer: DataLayer for feature engine.
            strategy_factory: Callable that returns a strategy callback.
                Signature: strategy_factory(train_bars) -> Callable
                The returned callback has signature: (bar, features, portfolio) -> list[Order]
        """
        self._config = config
        self._sentinel_stack = sentinel_stack
        self._data_layer = data_layer
        self._strategy_factory = strategy_factory

    def run(self, bars: list[Bar]) -> WalkForwardResults:
        """
        Execute full walk-forward validation over the bar dataset.

        The dataset is split into rolling windows:
          [train_bars][test_bars]
                  [step]
                     [train_bars][test_bars]
                             [step]
                                ...

        Returns WalkForwardResults with all window metrics.
        """
        cfg = self._config
        total = len(bars)
        min_required = cfg.train_bars + cfg.test_bars

        if total < min_required:
            logger.warning(
                "Insufficient data: %d bars < %d required (train+test)",
                total, min_required,
            )
            return WalkForwardResults(
                config=cfg,
                deployment_approved=False,
                deployment_reason=f"INSUFFICIENT_DATA: {total} bars < {min_required}",
            )

        # Compute window boundaries
        windows: list[WalkForwardWindow] = []
        all_oos_trades: list[BacktestTrade] = []
        all_oos_daily_returns: list[float] = []

        window_idx = 0
        start = 0

        while start + cfg.train_bars + cfg.test_bars <= total:
            train_start = start
            train_end = start + cfg.train_bars
            test_start = train_end
            test_end = min(test_start + cfg.test_bars, total)

            train_slice = bars[train_start:train_end]
            test_slice = bars[test_start:test_end]

            logger.info(
                "Window %d: train[%d:%d] test[%d:%d]",
                window_idx, train_start, train_end, test_start, test_end,
            )

            # Generate strategy from training data
            strategy_cb = self._strategy_factory(train_slice)

            # Run backtest on OOS test data
            bt_config = BacktestConfig(
                symbol=cfg.backtest_config.symbol,
                timeframe=cfg.backtest_config.timeframe,
                initial_capital=cfg.backtest_config.initial_capital,
                risk_per_trade=cfg.backtest_config.risk_per_trade,
                broker_config=cfg.backtest_config.broker_config,
                max_bars=None,
                warmup_bars=min(cfg.backtest_config.warmup_bars, len(test_slice) // 10),
                random_seed=cfg.backtest_config.random_seed + window_idx,
                enable_feature_engine=cfg.backtest_config.enable_feature_engine,
            )

            test_sentinel_stack = self._fresh_sentinel_stack()
            engine = BacktestEngine(bt_config, test_sentinel_stack, self._data_layer)
            engine.set_strategy(strategy_cb)
            results = engine.run(test_slice)

            # Compute metrics for this OOS window
            test_metrics = compute_metrics(
                trades=results.trades,
                equity_curve=results.equity_curve[1],  # equity values
                daily_returns=results.daily_returns,
                initial_capital=bt_config.initial_capital,
                total_bars=len(test_slice),
            )

            if test_metrics.total_trades < cfg.min_trades_per_window:
                logger.info(
                    "Window %d skipped: %d trades < %d minimum",
                    window_idx, test_metrics.total_trades, cfg.min_trades_per_window,
                )
                window_idx += 1
                start += cfg.step_bars
                continue

            # Also run backtest on train data for comparison
            train_sentinel_stack = self._fresh_sentinel_stack()
            train_engine = BacktestEngine(bt_config, train_sentinel_stack, self._data_layer)
            train_strategy = self._strategy_factory(train_slice)
            train_engine.set_strategy(train_strategy)
            train_results = train_engine.run(train_slice)

            train_metrics = compute_metrics(
                trades=train_results.trades,
                equity_curve=train_results.equity_curve[1],
                daily_returns=train_results.daily_returns,
                initial_capital=bt_config.initial_capital,
                total_bars=len(train_slice),
            )

            window_mc = None
            if results.trades:
                window_mc_engine = MonteCarloEngine(
                    MonteCarloConfig(
                        num_paths=cfg.monte_carlo_paths,
                        random_seed=cfg.backtest_config.random_seed + window_idx,
                    )
                )
                window_mc = window_mc_engine.run(
                    results.trades,
                    initial_capital=bt_config.initial_capital,
                )

            # Build window result
            wf_window = WalkForwardWindow(
                window_index=window_idx,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_bars_count=len(train_slice),
                test_bars_count=len(test_slice),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                test_trades=results.trades,
                test_equity_curve=results.equity_curve[1],
                mc_results=window_mc,
            )

            windows.append(wf_window)
            all_oos_trades.extend(results.trades)
            all_oos_daily_returns.extend(results.daily_returns)

            window_idx += 1
            start += cfg.step_bars

        # Combined OOS metrics across all windows
        combined_metrics = None
        if all_oos_trades:
            combined_equity = self._build_combined_equity(
                all_oos_trades, cfg.backtest_config.initial_capital,
            )
            combined_metrics = compute_metrics(
                trades=all_oos_trades,
                equity_curve=combined_equity,
                daily_returns=all_oos_daily_returns,
                initial_capital=cfg.backtest_config.initial_capital,
                total_bars=sum(w.test_bars_count for w in windows),
            )

        # Monte Carlo on combined OOS trades
        combined_mc = None
        if all_oos_trades:
            mc_engine = MonteCarloEngine(
                MonteCarloConfig(
                    num_paths=cfg.monte_carlo_paths,
                    random_seed=cfg.backtest_config.random_seed,
                )
            )
            combined_mc = mc_engine.run(
                all_oos_trades,
                initial_capital=cfg.backtest_config.initial_capital,
            )

        # Aggregate OOS Sharpe statistics
        oos_sharpes = [w.oos_sharpe for w in windows if w.test_metrics]
        avg_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        min_sharpe = float(np.min(oos_sharpes)) if oos_sharpes else 0.0
        max_sharpe = float(np.max(oos_sharpes)) if oos_sharpes else 0.0
        std_sharpe = float(np.std(oos_sharpes, ddof=1)) if len(oos_sharpes) > 1 else 0.0

        avg_ret = float(np.mean([w.oos_return_pct for w in windows])) if windows else 0.0
        median_ret = (
            float(np.median([w.oos_return_pct for w in windows])) if windows else 0.0
        )
        avg_mdd = float(np.mean([w.oos_mdd_pct for w in windows])) if windows else 0.0
        total_oos = sum(w.oos_trade_count for w in windows)
        profitable_windows = sum(1 for w in windows if w.oos_return_pct > 0.0)
        profitable_window_ratio = (
            profitable_windows / len(windows) if windows else 0.0
        )
        combined_profitability_score = (
            combined_metrics.profitability_score if combined_metrics else 0.0
        )

        oos_wr = 0.0
        if all_oos_trades:
            wins = sum(1 for t in all_oos_trades if t.net_pnl > 0)
            oos_wr = wins / len(all_oos_trades) * 100

        # Deployment decision
        approved, reason = self._deployment_check(
            windows, avg_sharpe, total_oos, combined_metrics, combined_mc, cfg,
        )

        return WalkForwardResults(
            config=cfg,
            windows=windows,
            total_oos_trades=total_oos,
            combined_oos_metrics=combined_metrics,
            combined_mc=combined_mc,
            avg_oos_sharpe=avg_sharpe,
            min_oos_sharpe=min_sharpe,
            max_oos_sharpe=max_sharpe,
            std_oos_sharpe=std_sharpe,
            avg_oos_return_pct=avg_ret,
            avg_oos_mdd_pct=avg_mdd,
            oos_win_rate_pct=oos_wr,
            profitable_windows=profitable_windows,
            profitable_window_ratio=profitable_window_ratio,
            median_oos_return_pct=median_ret,
            combined_profitability_score=combined_profitability_score,
            deployment_approved=approved,
            deployment_reason=reason,
        )

    def _fresh_sentinel_stack(self) -> dict:
        """
        Build an isolated SENTINEL stack for each fold run so state
        (open positions, drawdown, L3 flags) cannot leak across windows.
        """
        try:
            from aphelion.core.event_bus import EventBus
            from aphelion.core.clock import MarketClock
            from aphelion.risk.sentinel.circuit_breaker import CircuitBreaker
            from aphelion.risk.sentinel.core import SentinelCore
            from aphelion.risk.sentinel.execution.enforcer import ExecutionEnforcer
            from aphelion.risk.sentinel.position_sizer import PositionSizer
            from aphelion.risk.sentinel.validator import TradeValidator

            base_core = self._sentinel_stack["core"]
            clock = getattr(base_core, "_clock", None) or MarketClock()  # noqa: SLF001
            event_bus = EventBus()

            core = SentinelCore(event_bus, clock)
            validator = TradeValidator(core, clock)
            cb = CircuitBreaker(event_bus)
            enforcer = ExecutionEnforcer(validator, cb)
            sizer = PositionSizer()
            return {
                "core": core,
                "validator": validator,
                "cb": cb,
                "enforcer": enforcer,
                "sizer": sizer,
            }
        except Exception:
            # Fallback to provided stack if an external sentinel implementation differs.
            return self._sentinel_stack

    @staticmethod
    def _build_combined_equity(
        trades: list[BacktestTrade], initial_capital: float,
    ) -> list[float]:
        equity = initial_capital
        curve = [equity]
        for trade in sorted(trades, key=lambda t: t.exit_time):
            equity += trade.net_pnl
            curve.append(equity)
        return curve

    @staticmethod
    def _deployment_check(
        windows: list[WalkForwardWindow],
        avg_sharpe: float,
        total_trades: int,
        combined_metrics: Optional[BacktestMetrics],
        mc: Optional[MonteCarloResults],
        cfg: WalkForwardConfig,
    ) -> tuple[bool, str]:
        """
        Determine if walk-forward results meet deployment criteria.

        Spec requirements:
          - Minimum 12 OOS windows
          - Walk-forward Sharpe > 1.0
          - 500 trades minimum (for Phase 4+)
          - MC P5 MDD < 20%
        """
        reasons: list[str] = []

        if len(windows) < cfg.min_windows:
            reasons.append(
                f"INSUFFICIENT_WINDOWS: {len(windows)} < {cfg.min_windows}"
            )

        if avg_sharpe < 1.0:
            reasons.append(f"LOW_AVG_SHARPE: {avg_sharpe:.2f} < 1.0")

        if total_trades < 500:
            reasons.append(f"TOO_FEW_TRADES: {total_trades} < 500")

        if not combined_metrics:
            reasons.append("NO_COMBINED_OOS_METRICS")
        else:
            if combined_metrics.net_profit <= 0:
                reasons.append(
                    f"NON_POSITIVE_OOS_NET_PROFIT: {combined_metrics.net_profit:.2f} <= 0"
                )
            if combined_metrics.profit_factor < 1.3:
                reasons.append(
                    f"LOW_OOS_PF: {combined_metrics.profit_factor:.2f} < 1.3"
                )
            if combined_metrics.expectancy_dollars <= 0:
                reasons.append(
                    f"NEG_OOS_EXPECTANCY: {combined_metrics.expectancy_dollars:.2f} <= 0"
                )
            if combined_metrics.return_over_max_drawdown < 1.2:
                reasons.append(
                    "LOW_OOS_RETURN_TO_DRAWDOWN: "
                    f"{combined_metrics.return_over_max_drawdown:.2f} < 1.20"
                )
            if combined_metrics.profitable_month_ratio < 55.0:
                reasons.append(
                    "LOW_OOS_MONTHLY_CONSISTENCY: "
                    f"{combined_metrics.profitable_month_ratio:.1f}% < 55%"
                )
            if combined_metrics.profitability_score < 0.50:
                reasons.append(
                    "LOW_OOS_PROFITABILITY_SCORE: "
                    f"{combined_metrics.profitability_score:.2f} < 0.50"
                )

        # Check MC risk bounds
        if mc and mc.p95_mdd > 20.0:
            reasons.append(f"HIGH_MC_MDD_P95: {mc.p95_mdd:.1f}% > 20%")

        if mc and mc.ruin_probability > 0.05:
            reasons.append(f"HIGH_RUIN_PROB: {mc.ruin_probability:.2%} > 5%")

        # Check consistency — no window should have deeply negative Sharpe
        negative_windows = sum(1 for w in windows if w.oos_sharpe < -0.5)
        if negative_windows > len(windows) * 0.25:
            reasons.append(
                f"INCONSISTENT: {negative_windows}/{len(windows)} windows have Sharpe < -0.5"
            )
        profitable_windows = sum(1 for w in windows if w.oos_return_pct > 0.0)
        profitable_ratio = profitable_windows / len(windows) if windows else 0.0
        if profitable_ratio < 0.60:
            reasons.append(
                f"LOW_PROFITABLE_WINDOWS: {profitable_windows}/{len(windows)} < 60%"
            )

        if reasons:
            return False, " | ".join(reasons)
        return True, "APPROVED: Walk-forward validation passed"
