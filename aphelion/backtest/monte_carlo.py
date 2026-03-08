"""
APHELION Monte Carlo Simulation Engine
500-path equity simulation with bootstrap resampling, sequence-risk support,
drawdown distribution, and ruin probability estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aphelion.backtest.metrics import max_drawdown
from aphelion.backtest.order import BacktestTrade


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    num_paths: int = 500
    random_seed: int = 42
    confidence_levels: tuple[int, int, int] = (5, 50, 95)
    resample_with_replacement: bool = True
    include_sequence_risk: bool = True


@dataclass
class MonteCarloResults:
    """Results from a Monte Carlo simulation run."""

    num_paths: int = 0
    num_trades: int = 0
    initial_capital: float = 0.0

    p5_equity: list[float] = field(default_factory=list)
    p50_equity: list[float] = field(default_factory=list)
    p95_equity: list[float] = field(default_factory=list)

    final_equities: list[float] = field(default_factory=list)
    p5_final: float = 0.0
    p50_final: float = 0.0
    p95_final: float = 0.0
    mean_final: float = 0.0

    max_drawdowns: list[float] = field(default_factory=list)
    p5_mdd: float = 0.0
    p50_mdd: float = 0.0
    p95_mdd: float = 0.0
    worst_mdd: float = 0.0

    ruin_probability: float = 0.0
    ruin_threshold_pct: float = 50.0

    probability_of_profit: float = 0.0
    mean_return_pct: float = 0.0
    median_return_pct: float = 0.0
    std_return_pct: float = 0.0
    expected_shortfall_p5_pct: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    profit_factors: list[float] = field(default_factory=list)
    p5_profit_factor: float = 0.0
    p50_profit_factor: float = 0.0
    p95_profit_factor: float = 0.0

    @property
    def deployment_safe(self) -> bool:
        return (
            self.p5_mdd < 20.0
            and self.ruin_probability < 0.05
            and self.probability_of_profit >= 0.55
        )

    def to_dict(self) -> dict:
        return {
            "num_paths": self.num_paths,
            "num_trades": self.num_trades,
            "initial_capital": self.initial_capital,
            "p5_equity": self.p5_equity,
            "p50_equity": self.p50_equity,
            "p95_equity": self.p95_equity,
            "final_equities": self.final_equities,
            "p5_final": self.p5_final,
            "p50_final": self.p50_final,
            "p95_final": self.p95_final,
            "mean_final": self.mean_final,
            "max_drawdowns": self.max_drawdowns,
            "p5_mdd": self.p5_mdd,
            "p50_mdd": self.p50_mdd,
            "p95_mdd": self.p95_mdd,
            "worst_mdd": self.worst_mdd,
            "ruin_probability": self.ruin_probability,
            "ruin_threshold_pct": self.ruin_threshold_pct,
            "probability_of_profit": self.probability_of_profit,
            "mean_return_pct": self.mean_return_pct,
            "median_return_pct": self.median_return_pct,
            "std_return_pct": self.std_return_pct,
            "expected_shortfall_p5_pct": self.expected_shortfall_p5_pct,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "profit_factors": self.profit_factors,
            "p5_profit_factor": self.p5_profit_factor,
            "p50_profit_factor": self.p50_profit_factor,
            "p95_profit_factor": self.p95_profit_factor,
        }


class MonteCarloEngine:
    """
    Monte Carlo path simulator for strategy risk.
    Uses bootstrap resampling and optional block sampling to preserve streak risk.
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self._config = config or MonteCarloConfig()
        self._rng = np.random.default_rng(self._config.random_seed)

    def run(
        self,
        trades: list[BacktestTrade],
        initial_capital: float = 10_000.0,
        ruin_threshold_pct: float = 50.0,
    ) -> MonteCarloResults:
        n_trades = len(trades)
        n_paths = self._config.num_paths

        if n_trades == 0:
            return MonteCarloResults(
                num_paths=n_paths,
                num_trades=0,
                initial_capital=initial_capital,
                p5_final=initial_capital,
                p50_final=initial_capital,
                p95_final=initial_capital,
                mean_final=initial_capital,
            )

        pnls = np.array([trade.net_pnl for trade in trades], dtype=np.float64)
        all_equity = np.zeros((n_paths, n_trades + 1), dtype=np.float64)
        all_equity[:, 0] = initial_capital
        path_profit_factors = np.zeros(n_paths, dtype=np.float64)

        for path_i in range(n_paths):
            sampled = self._sample_path(pnls)
            wins = float(np.sum(sampled[sampled > 0]))
            losses = float(abs(np.sum(sampled[sampled < 0])))
            if losses == 0:
                pf = 10.0 if wins > 0 else 0.0
            else:
                pf = wins / losses
            path_profit_factors[path_i] = min(pf, 10.0)
            equity = initial_capital
            for step_i, pnl in enumerate(sampled, start=1):
                equity = max(0.0, equity + float(pnl))
                all_equity[path_i, step_i] = equity

        p5_curve = np.percentile(all_equity, 5, axis=0).tolist()
        p50_curve = np.percentile(all_equity, 50, axis=0).tolist()
        p95_curve = np.percentile(all_equity, 95, axis=0).tolist()

        finals = all_equity[:, -1]
        p5_f = float(np.percentile(finals, 5))
        p50_f = float(np.percentile(finals, 50))
        p95_f = float(np.percentile(finals, 95))
        mean_f = float(np.mean(finals))

        mdds = np.zeros(n_paths, dtype=np.float64)
        for path_i in range(n_paths):
            dd, _, _ = max_drawdown(all_equity[path_i].tolist())
            mdds[path_i] = dd * 100.0

        p5_mdd = float(np.percentile(mdds, 5))
        p50_mdd = float(np.percentile(mdds, 50))
        p95_mdd = float(np.percentile(mdds, 95))
        worst_mdd = float(np.max(mdds))

        ruin_level = initial_capital * (1.0 - ruin_threshold_pct / 100.0)
        ruin_prob = float(np.mean(np.min(all_equity, axis=1) <= ruin_level))
        prob_profit = float(np.mean(finals > initial_capital))

        returns_pct = ((finals - initial_capital) / initial_capital) * 100.0
        mean_ret = float(np.mean(returns_pct))
        median_ret = float(np.percentile(returns_pct, 50))
        std_ret = float(np.std(returns_pct, ddof=1)) if n_paths > 1 else 0.0
        p5_cutoff = float(np.percentile(returns_pct, 5))
        tail = returns_pct[returns_pct <= p5_cutoff]
        es_p5 = float(np.mean(tail)) if len(tail) > 0 else p5_cutoff
        if std_ret > 0:
            z = (returns_pct - mean_ret) / std_ret
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4))
        else:
            skew = 0.0
            kurt = 3.0
        p5_pf = float(np.percentile(path_profit_factors, 5))
        p50_pf = float(np.percentile(path_profit_factors, 50))
        p95_pf = float(np.percentile(path_profit_factors, 95))

        return MonteCarloResults(
            num_paths=n_paths,
            num_trades=n_trades,
            initial_capital=initial_capital,
            p5_equity=p5_curve,
            p50_equity=p50_curve,
            p95_equity=p95_curve,
            final_equities=finals.tolist(),
            p5_final=p5_f,
            p50_final=p50_f,
            p95_final=p95_f,
            mean_final=mean_f,
            max_drawdowns=mdds.tolist(),
            p5_mdd=p5_mdd,
            p50_mdd=p50_mdd,
            p95_mdd=p95_mdd,
            worst_mdd=worst_mdd,
            ruin_probability=ruin_prob,
            ruin_threshold_pct=ruin_threshold_pct,
            probability_of_profit=prob_profit,
            mean_return_pct=mean_ret,
            median_return_pct=median_ret,
            std_return_pct=std_ret,
            expected_shortfall_p5_pct=es_p5,
            skewness=skew,
            kurtosis=kurt,
            profit_factors=path_profit_factors.tolist(),
            p5_profit_factor=p5_pf,
            p50_profit_factor=p50_pf,
            p95_profit_factor=p95_pf,
        )

    def stress_test(
        self,
        trades: list[BacktestTrade],
        initial_capital: float = 10_000.0,
        ruin_threshold_pct: float = 50.0,
        adverse_factor: float = 1.5,
    ) -> MonteCarloResults:
        """
        Stress environment simulation by amplifying losses.
        Example: adverse_factor=1.5 means every loss is 50% worse.
        """
        factor = max(1.0, float(adverse_factor))
        stressed: list[BacktestTrade] = []
        for trade in trades:
            if trade.net_pnl < 0:
                stressed.append(
                    BacktestTrade(
                        trade_id=trade.trade_id,
                        symbol=trade.symbol,
                        direction=trade.direction,
                        entry_price=trade.entry_price,
                        exit_price=trade.exit_price,
                        size_lots=trade.size_lots,
                        size_pct=trade.size_pct,
                        stop_loss=trade.stop_loss,
                        take_profit=trade.take_profit,
                        entry_time=trade.entry_time,
                        exit_time=trade.exit_time,
                        gross_pnl=trade.gross_pnl * factor,
                        commission=trade.commission,
                        net_pnl=trade.net_pnl * factor,
                        exit_reason=trade.exit_reason,
                        bars_held=trade.bars_held,
                        proposed_by=trade.proposed_by,
                        entry_bar_index=trade.entry_bar_index,
                        exit_bar_index=trade.exit_bar_index,
                    )
                )
            else:
                stressed.append(trade)

        return self.run(
            stressed,
            initial_capital=initial_capital,
            ruin_threshold_pct=ruin_threshold_pct,
        )

    def bootstrap_sharpe(
        self, daily_returns: list[float], n_bootstrap: int = 10_000,
    ) -> dict:
        """
        Stationary block bootstrap confidence interval for Sharpe ratio.
        """
        if not daily_returns:
            return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p95": 0.0}

        data = np.array(daily_returns, dtype=np.float64)
        n = len(data)
        block_size = max(3, n // 20)
        values: list[float] = []

        for _ in range(int(n_bootstrap)):
            sample = self._stationary_block_bootstrap(data, n, block_size)
            std = float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0
            if std == 0:
                values.append(0.0)
            else:
                values.append(float(np.sqrt(252.0) * float(np.mean(sample)) / std))

        arr = np.array(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }

    def _sample_path(self, pnls: np.ndarray) -> np.ndarray:
        n = len(pnls)
        if not self._config.resample_with_replacement:
            return self._rng.permutation(pnls)

        if self._config.include_sequence_risk and n >= 5:
            return self._block_bootstrap(pnls)

        return self._rng.choice(pnls, size=n, replace=True)

    def _block_bootstrap(self, pnls: np.ndarray) -> np.ndarray:
        n = len(pnls)
        block_size = max(3, n // 20)
        sampled = np.empty(n, dtype=np.float64)
        out_i = 0
        while out_i < n:
            start = int(self._rng.integers(0, n))
            take = min(block_size, n - out_i)
            for j in range(take):
                sampled[out_i + j] = pnls[(start + j) % n]
            out_i += take
        return sampled

    def _stationary_block_bootstrap(
        self, series: np.ndarray, sample_size: int, block_size: int,
    ) -> np.ndarray:
        sampled = np.empty(sample_size, dtype=np.float64)
        n = len(series)
        i = 0
        while i < sample_size:
            start = int(self._rng.integers(0, n))
            take = min(block_size, sample_size - i)
            for j in range(take):
                sampled[i + j] = series[(start + j) % n]
            i += take
        return sampled
