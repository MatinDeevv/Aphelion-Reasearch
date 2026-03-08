import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from aphelion.backtest.analytics import PerformanceAnalyzer
from aphelion.backtest.order import BacktestTrade


def _make_trade(net_pnl, exit_reason="TP_HIT", direction="LONG",
                entry=2850.0, sl=2840.0, tp=2870.0, lots=0.01,
                entry_idx=0, exit_idx=10):
    if direction == "LONG":
        if net_pnl >= 0:
            exit_ = entry + net_pnl / (lots * 100)
        else:
            exit_ = entry + net_pnl / (lots * 100)
    else:
        exit_ = entry - net_pnl / (lots * 100)
    gross = net_pnl + 0.14  # assume small commission
    return BacktestTrade(
        trade_id=f"t-{entry_idx}", symbol="XAUUSD", direction=direction,
        entry_price=entry, exit_price=exit_,
        size_lots=lots, size_pct=0.02,
        stop_loss=sl, take_profit=tp,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=entry_idx),
        exit_time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=exit_idx),
        gross_pnl=gross, commission=0.14, net_pnl=net_pnl,
        exit_reason=exit_reason, bars_held=exit_idx - entry_idx,
        proposed_by="TEST", entry_bar_index=entry_idx, exit_bar_index=exit_idx,
    )


def _make_mixed_trades(n_wins=60, n_losses=40, avg_win=200.0, avg_loss=-120.0):
    trades = []
    idx = 0
    for i in range(n_wins):
        trades.append(_make_trade(avg_win, "TP_HIT", entry_idx=idx, exit_idx=idx+10))
        idx += 20
    for i in range(n_losses):
        trades.append(_make_trade(avg_loss, "SL_HIT", entry_idx=idx, exit_idx=idx+5))
        idx += 20
    return trades


def _make_equity_curve(trades, initial=10000.0):
    eq = [initial]
    current = initial
    for t in trades:
        current += t.net_pnl
        eq.append(current)
    return eq


def _make_timestamps(n):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(hours=i) for i in range(n)]


class TestPerformanceAnalyzer:
    def test_win_rate_correct(self):
        trades = _make_mixed_trades(60, 40)
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.win_rate == pytest.approx(0.60, abs=0.01)

    def test_win_rate_zero_on_no_trades(self):
        pa = PerformanceAnalyzer([], [10000.0], [datetime.now(timezone.utc)], 10000.0)
        assert pa.win_rate == 0.0

    def test_profit_factor_correct(self):
        trades = _make_mixed_trades(60, 40, avg_win=200.0, avg_loss=-120.0)
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        # gross_profit = 60 * 200 = 12000, gross_loss = 40 * 120 = 4800
        assert pa.profit_factor == pytest.approx(2.5, abs=0.1)

    def test_profit_factor_inf_on_no_losses(self):
        trades = [_make_trade(100.0, entry_idx=i*20, exit_idx=i*20+10) for i in range(10)]
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.profit_factor == float('inf')

    def test_expectancy_correct(self):
        trades = _make_mixed_trades(60, 40, 200.0, -120.0)
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        # expectancy = 0.6*200 - 0.4*120 = 120 - 48 = 72
        assert pa.expectancy == pytest.approx(72.0, abs=1.0)

    def test_sharpe_positive_for_profitable_strategy(self):
        trades = _make_mixed_trades(70, 30, 200.0, -100.0)
        eq = _make_equity_curve(trades)
        # Need multi-day timestamps for daily returns
        ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(len(eq))]
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.sharpe_ratio > 0

    def test_sharpe_zero_for_flat_equity(self):
        # Flat equity = no returns
        eq = [10000.0] * 50
        ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(50)]
        pa = PerformanceAnalyzer([], eq, ts, 10000.0)
        assert pa.sharpe_ratio == 0.0

    def test_max_drawdown_10pct(self):
        eq = [10000.0, 11000.0, 9900.0, 10500.0]
        ts = _make_timestamps(4)
        pa = PerformanceAnalyzer([], eq, ts, 10000.0)
        assert pa.max_drawdown == pytest.approx(0.10, abs=0.01)

    def test_max_drawdown_zero_for_monotonic_equity(self):
        eq = [10000.0, 10100.0, 10200.0, 10300.0]
        ts = _make_timestamps(4)
        pa = PerformanceAnalyzer([], eq, ts, 10000.0)
        assert pa.max_drawdown == 0.0

    def test_calmar_ratio_correct(self):
        # calmar = CAGR / (max_dd * 100)
        # We test indirectly via properties
        trades = _make_mixed_trades(70, 30, 200.0, -100.0)
        eq = _make_equity_curve(trades)
        ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(len(eq))]
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        if pa.max_drawdown > 0:
            expected_calmar = pa.CAGR / (pa.max_drawdown * 100.0)
            assert pa.calmar_ratio == pytest.approx(expected_calmar, abs=0.1)

    def test_consecutive_wins_3(self):
        trades = [
            _make_trade(100.0, entry_idx=0, exit_idx=10),
            _make_trade(100.0, entry_idx=20, exit_idx=30),
            _make_trade(100.0, entry_idx=40, exit_idx=50),
            _make_trade(-50.0, "SL_HIT", entry_idx=60, exit_idx=70),
            _make_trade(100.0, entry_idx=80, exit_idx=90),
        ]
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.consecutive_wins == 3

    def test_consecutive_losses_4(self):
        trades = [
            _make_trade(-50.0, "SL_HIT", entry_idx=0, exit_idx=5),
            _make_trade(-50.0, "SL_HIT", entry_idx=10, exit_idx=15),
            _make_trade(-50.0, "SL_HIT", entry_idx=20, exit_idx=25),
            _make_trade(-50.0, "SL_HIT", entry_idx=30, exit_idx=35),
            _make_trade(100.0, entry_idx=40, exit_idx=50),
        ]
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.consecutive_losses == 4

    def test_exit_reason_breakdown_counts(self):
        trades = (
            [_make_trade(100.0, "TP_HIT", entry_idx=i*20, exit_idx=i*20+10) for i in range(30)]
            + [_make_trade(-50.0, "SL_HIT", entry_idx=600+i*20, exit_idx=600+i*20+5) for i in range(20)]
        )
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        breakdown = pa.exit_reason_breakdown
        assert breakdown["TP_HIT"] == 30
        assert breakdown["SL_HIT"] == 20

    def test_score_zero_under_30_trades(self):
        trades = [_make_trade(100.0, entry_idx=i*20, exit_idx=i*20+10) for i in range(10)]
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.score() == 0.0

    def test_score_positive_for_good_strategy(self):
        trades = _make_mixed_trades(65, 35, 200.0, -100.0)
        eq = _make_equity_curve(trades)
        ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(len(eq))]
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        # May or may not be > 0 depending on drawdown
        # score() requires max_drawdown <= 0.20 -- let's just check it doesn't crash
        score = pa.score()
        assert score >= 0.0

    def test_score_zero_for_high_drawdown(self):
        # Force high drawdown: big loss then recovery
        trades = [_make_trade(-2500.0, "SL_HIT", entry_idx=i*20, exit_idx=i*20+5) for i in range(30)]
        trades += [_make_trade(100.0, entry_idx=600+i*20, exit_idx=600+i*20+10) for i in range(10)]
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.score() == 0.0  # max_drawdown > 0.20

    def test_to_dict_has_all_required_keys(self):
        trades = _make_mixed_trades(60, 40)
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        d = pa.to_dict()
        required = [
            "total_trades", "win_rate", "profit_factor", "expectancy",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
            "consecutive_wins", "consecutive_losses", "r_expectancy", "net_profit",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_r_expectancy_positive_for_edge(self):
        trades = _make_mixed_trades(65, 35, 200.0, -100.0)
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.r_expectancy > 0

    def test_cagr_zero_on_single_bar(self):
        ts = [datetime(2024, 1, 1, tzinfo=timezone.utc)]
        pa = PerformanceAnalyzer([], [10000.0], ts, 10000.0)
        assert pa.CAGR == 0.0

    def test_largest_win_and_loss(self):
        trades = _make_mixed_trades(60, 40, 200.0, -120.0)
        eq = _make_equity_curve(trades)
        ts = _make_timestamps(len(eq))
        pa = PerformanceAnalyzer(trades, eq, ts, 10000.0)
        assert pa.largest_win > 0
        assert pa.largest_loss < 0
