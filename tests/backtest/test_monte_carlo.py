import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from aphelion.backtest.monte_carlo import MonteCarloEngine, MonteCarloConfig, MonteCarloResults
from aphelion.backtest.order import BacktestTrade


def _make_trade(net_pnl, idx=0):
    return BacktestTrade(
        trade_id=f"mc-{idx}", symbol="XAUUSD", direction="LONG",
        entry_price=2850.0, exit_price=2850.0 + net_pnl / 1.0,
        size_lots=0.01, size_pct=0.02,
        stop_loss=2840.0, take_profit=2870.0,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=idx*20),
        exit_time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=idx*20+10),
        gross_pnl=net_pnl + 0.14, commission=0.14, net_pnl=net_pnl,
        exit_reason="TP_HIT" if net_pnl > 0 else "SL_HIT",
        bars_held=10, proposed_by="TEST",
        entry_bar_index=idx*20, exit_bar_index=idx*20+10,
    )


def _good_trades(n=50):
    """65% win rate, avg_win=200, avg_loss=-100"""
    rng = np.random.default_rng(42)
    trades = []
    for i in range(n):
        if rng.random() < 0.65:
            trades.append(_make_trade(200.0, i))
        else:
            trades.append(_make_trade(-100.0, i))
    return trades


def _bad_trades(n=50):
    """35% win rate, avg_win=100, avg_loss=-200"""
    rng = np.random.default_rng(42)
    trades = []
    for i in range(n):
        if rng.random() < 0.35:
            trades.append(_make_trade(100.0, i))
        else:
            trades.append(_make_trade(-200.0, i))
    return trades


class TestMonteCarlo:
    def test_raises_on_fewer_than_10_trades(self):
        mc = MonteCarloEngine()
        trades = [_make_trade(100.0, i) for i in range(5)]
        with pytest.raises(ValueError, match="at least 10"):
            mc.run(trades)

    def test_probability_of_profit_high_for_good_strategy(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        result = mc.run(_good_trades(50), initial_capital=10000.0)
        assert result.probability_of_profit > 0.75

    def test_probability_of_ruin_low_for_good_strategy(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        result = mc.run(_good_trades(50), initial_capital=10000.0)
        assert result.ruin_probability < 0.15

    def test_bad_strategy_has_low_probability_of_profit(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        result = mc.run(_bad_trades(50), initial_capital=10000.0)
        assert result.probability_of_profit < 0.50

    def test_percentile_ordering(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        result = mc.run(_good_trades(50), initial_capital=10000.0)
        assert result.p5_final <= result.p50_final <= result.p95_final

    def test_worst_case_below_expected(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        result = mc.run(_good_trades(50), initial_capital=10000.0)
        worst = min(result.final_equities)
        assert worst < result.mean_final

    def test_stress_test_has_higher_ruin_than_normal(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        trades = _good_trades(50)
        normal = mc.run(trades, initial_capital=10000.0)
        mc2 = MonteCarloEngine(MonteCarloConfig(num_paths=200, random_seed=42))
        stressed = mc2.stress_test(trades, initial_capital=10000.0, adverse_factor=1.5)
        assert stressed.ruin_probability >= normal.ruin_probability

    def test_reproducible_same_seed(self):
        trades = _good_trades(50)
        mc1 = MonteCarloEngine(MonteCarloConfig(num_paths=100, random_seed=42))
        r1 = mc1.run(trades, initial_capital=10000.0)
        mc2 = MonteCarloEngine(MonteCarloConfig(num_paths=100, random_seed=42))
        r2 = mc2.run(trades, initial_capital=10000.0)
        assert r1.probability_of_profit == r2.probability_of_profit

    def test_different_seeds_differ(self):
        trades = _good_trades(50)
        mc1 = MonteCarloEngine(MonteCarloConfig(num_paths=100, random_seed=42))
        r1 = mc1.run(trades, initial_capital=10000.0)
        mc2 = MonteCarloEngine(MonteCarloConfig(num_paths=100, random_seed=99))
        r2 = mc2.run(trades, initial_capital=10000.0)
        # Very unlikely to be exactly equal with different seeds
        assert r1.mean_final != r2.mean_final or r1.p5_final != r2.p5_final

    def test_equity_never_negative_in_simulation(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=100, random_seed=42))
        result = mc.run(_bad_trades(50), initial_capital=10000.0)
        for eq in result.final_equities:
            assert eq >= 0.0

    def test_to_dict_serializable(self):
        mc = MonteCarloEngine(MonteCarloConfig(num_paths=50, random_seed=42))
        result = mc.run(_good_trades(20), initial_capital=10000.0)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "probability_of_profit" in d
        assert "ruin_probability" in d

    def test_bootstrap_sharpe_confidence_interval(self):
        rng = np.random.default_rng(42)
        daily_returns = rng.normal(0.001, 0.01, size=252).tolist()
        mc = MonteCarloEngine(MonteCarloConfig(random_seed=42))
        result = mc.bootstrap_sharpe(daily_returns, n_bootstrap=500)
        assert "p05" in result
        assert "p95" in result
        assert "mean" in result
        assert result["p05"] < result["mean"] < result["p95"]
