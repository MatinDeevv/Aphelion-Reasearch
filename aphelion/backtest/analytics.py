"""Backtest performance analytics for institutional strategy evaluation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from math import sqrt
from statistics import mean

import numpy as np

from aphelion.backtest.order import BacktestTrade


class PerformanceAnalyzer:
    """Computes trading performance and risk metrics from closed trades."""

    def __init__(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        timestamps: list[datetime],
        initial_capital: float,
        risk_free_rate: float = 0.05,
    ):
        self._trades = trades
        self._equity_curve = equity_curve
        self._timestamps = timestamps
        self._initial_capital = float(initial_capital)
        self._risk_free_rate = float(risk_free_rate)

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for trade in self._trades if trade.net_pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for trade in self._trades if trade.net_pnl < 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        wins = [trade.net_pnl for trade in self._trades if trade.net_pnl > 0]
        return float(mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(trade.net_pnl) for trade in self._trades if trade.net_pnl < 0]
        return float(mean(losses)) if losses else 0.0

    @property
    def payoff_ratio(self) -> float:
        if self.avg_loss == 0.0:
            return float("inf") if self.avg_win > 0.0 else 0.0
        return self.avg_win / self.avg_loss

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf")
        return self.gross_profit / self.gross_loss

    @property
    def expectancy(self) -> float:
        return (self.win_rate * self.avg_win) - ((1.0 - self.win_rate) * self.avg_loss)

    @property
    def expectancy_per_r(self) -> float:
        if self.avg_loss <= 0:
            return 0.0
        return self.expectancy / self.avg_loss

    @property
    def gross_profit(self) -> float:
        return float(sum(trade.net_pnl for trade in self._trades if trade.net_pnl > 0))

    @property
    def gross_loss(self) -> float:
        return float(sum(abs(trade.net_pnl) for trade in self._trades if trade.net_pnl < 0))

    @property
    def net_profit(self) -> float:
        return float(sum(trade.net_pnl for trade in self._trades))

    @property
    def total_commission(self) -> float:
        return float(sum(trade.commission for trade in self._trades))

    @property
    def total_return_pct(self) -> float:
        if self._initial_capital <= 0:
            return 0.0
        final_equity = self._final_equity()
        return ((final_equity - self._initial_capital) / self._initial_capital) * 100.0

    @property
    def CAGR(self) -> float:
        if len(self._timestamps) >= 2:
            trading_days = (self._timestamps[-1] - self._timestamps[0]).days
        else:
            trading_days = 1
        years = trading_days / 365.25
        if years <= 0:
            return 0.0

        final_equity = self._final_equity()
        if self._initial_capital <= 0 or final_equity <= 0:
            return 0.0
        return ((final_equity / self._initial_capital) ** (1.0 / years) - 1.0) * 100.0

    @property
    def max_drawdown(self) -> float:
        peak = self._initial_capital
        max_dd = 0.0
        for equity in self._equity_series():
            if equity > peak:
                peak = equity
            if peak > 0:
                drawdown = (peak - equity) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
        return max_dd

    @property
    def max_drawdown_duration_bars(self) -> int:
        peak = self._initial_capital
        current = 0
        longest = 0
        for equity in self._equity_series():
            if equity > peak:
                peak = equity
            if equity < peak:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return longest

    @property
    def avg_drawdown(self) -> float:
        peak = self._initial_capital
        drawdowns: list[float] = []
        for equity in self._equity_series():
            if equity > peak:
                peak = equity
            if peak > 0:
                drawdown = (peak - equity) / peak
                if drawdown > 0:
                    drawdowns.append(drawdown)
        return float(mean(drawdowns)) if drawdowns else 0.0

    @property
    def daily_returns(self) -> list[float]:
        n = min(len(self._equity_curve), len(self._timestamps))
        if n < 2:
            return []

        day_last_equity: dict[datetime.date, float] = {}
        for idx in range(n):
            day_last_equity[self._timestamps[idx].date()] = float(self._equity_curve[idx])

        daily_closes = [day_last_equity[day] for day in sorted(day_last_equity.keys())]
        if len(daily_closes) < 2:
            return []

        returns: list[float] = []
        for prev, curr in zip(daily_closes[:-1], daily_closes[1:]):
            if prev == 0:
                continue
            returns.append((curr / prev) - 1.0)
        return returns

    @property
    def sharpe_ratio(self) -> float:
        returns = self.daily_returns
        if not returns:
            return 0.0

        rf_daily = self._risk_free_rate / 252.0
        excess = np.array([ret - rf_daily for ret in returns], dtype=float)
        std = float(np.std(excess))
        if std < 1e-10:  # FIXED: floating-point epsilon guard
            return 0.0
        return float(sqrt(252.0) * float(np.mean(excess)) / std)

    @property
    def sortino_ratio(self) -> float:
        returns = self.daily_returns
        if not returns:
            return 0.0

        rf_daily = self._risk_free_rate / 252.0
        excess = np.array([ret - rf_daily for ret in returns], dtype=float)
        downside = np.array([ret for ret in returns if ret < 0], dtype=float)
        downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
        if downside_std < 1e-10:  # FIXED: floating-point epsilon guard
            return 0.0
        return float(sqrt(252.0) * float(np.mean(excess)) / downside_std)

    @property
    def calmar_ratio(self) -> float:
        dd = self.max_drawdown
        if dd == 0:
            return float("inf")
        return self.CAGR / (dd * 100.0)

    @property
    def mar_ratio(self) -> float:
        return self.calmar_ratio

    @property
    def return_over_drawdown(self) -> float:
        dd = self.max_drawdown
        if dd == 0.0:
            return float("inf") if self.total_return_pct > 0.0 else 0.0
        return self.total_return_pct / (dd * 100.0)

    @property
    def avg_trade_duration_bars(self) -> float:
        if not self._trades:
            return 0.0
        return float(mean(trade.bars_held for trade in self._trades))

    @property
    def avg_bars_between_trades(self) -> float:
        if len(self._trades) < 2:
            return 0.0
        ordered = sorted(self._trades, key=lambda t: t.entry_bar_index)
        gaps = [
            max(0, ordered[i].entry_bar_index - ordered[i - 1].exit_bar_index)
            for i in range(1, len(ordered))
        ]
        return float(mean(gaps)) if gaps else 0.0

    @property
    def consecutive_wins(self) -> int:
        best = 0
        run = 0
        for trade in self._trades:
            if trade.net_pnl > 0:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    @property
    def consecutive_losses(self) -> int:
        best = 0
        run = 0
        for trade in self._trades:
            if trade.net_pnl < 0:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best

    @property
    def longest_losing_streak_days(self) -> int:
        if not self._trades:
            return 0

        by_day: dict[datetime.date, float] = {}
        for trade in self._trades:
            day = trade.exit_time.date()
            by_day[day] = by_day.get(day, 0.0) + trade.net_pnl

        streak = 0
        longest = 0
        for day in sorted(by_day.keys()):
            if by_day[day] < 0:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        return longest

    @property
    def r_multiples(self) -> list[float]:
        return [float(trade.r_multiple) for trade in self._trades]

    @property
    def r_expectancy(self) -> float:
        values = self.r_multiples
        return float(mean(values)) if values else 0.0

    @property
    def largest_win(self) -> float:
        if not self._trades:
            return 0.0
        return float(max(trade.net_pnl for trade in self._trades))

    @property
    def largest_loss(self) -> float:
        if not self._trades:
            return 0.0
        return float(min(trade.net_pnl for trade in self._trades))

    @property
    def exit_reason_breakdown(self) -> dict[str, int]:
        return dict(Counter(trade.exit_reason for trade in self._trades))

    @property
    def ulcer_index(self) -> float:
        """Ulcer Index: measures depth and duration of drawdowns.
        Lower is better. Very sensitive to prolonged drawdowns."""
        series = self._equity_series()
        if len(series) < 2:
            return 0.0
        peak = series[0]
        sum_sq = 0.0
        for eq in series:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd_pct = ((peak - eq) / peak) * 100.0
                sum_sq += dd_pct ** 2
        return float(sqrt(sum_sq / len(series)))

    @property
    def ulcer_performance_index(self) -> float:
        """Return / Ulcer Index — risk-adjusted performance metric."""
        ui = self.ulcer_index
        if ui < 1e-10:
            return 0.0
        return self.total_return_pct / ui

    @property
    def tail_ratio(self) -> float:
        """95th percentile return / abs(5th percentile return).
        > 1 means right tail is fatter (desirable)."""
        returns = self.daily_returns
        if len(returns) < 20:
            return 1.0
        arr = np.array(returns)
        p95 = float(np.percentile(arr, 95))
        p5 = abs(float(np.percentile(arr, 5)))
        if p5 < 1e-10:
            return float("inf") if p95 > 0 else 1.0
        return p95 / p5

    @property
    def session_performance(self) -> dict[str, dict]:
        """PnL breakdown by trading session (if trades have session info)."""
        sessions: dict[str, list[float]] = {}
        for trade in self._trades:
            session = getattr(trade, "session", "UNKNOWN")
            sessions.setdefault(session, []).append(trade.net_pnl)
        result = {}
        for session, pnls in sessions.items():
            result[session] = {
                "trades": len(pnls),
                "net_pnl": sum(pnls),
                "avg_pnl": float(mean(pnls)) if pnls else 0.0,
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
            }
        return result

    @property
    def profitability_score(self) -> float:
        if self.total_trades < 30 or self.net_profit <= 0.0 or self.expectancy <= 0.0:
            return 0.0
        pf_term = min(max(self.profit_factor, 0.0), 5.0) / 5.0
        exp_term = min(max(self.expectancy_per_r, 0.0), 1.5) / 1.5
        rr_term = min(max(self.return_over_drawdown, 0.0), 3.0) / 3.0
        win_term = min(max(self.win_rate, 0.0), 1.0)
        consistency = 1.0 - (self.consecutive_losses / max(self.total_trades, 1))
        return (
            0.30 * pf_term
            + 0.25 * exp_term
            + 0.20 * rr_term
            + 0.15 * win_term
            + 0.10 * consistency
        )

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "payoff_ratio": self.payoff_ratio,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "expectancy_per_r": self.expectancy_per_r,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "total_commission": self.total_commission,
            "total_return_pct": self.total_return_pct,
            "CAGR": self.CAGR,
            "cagr": self.CAGR,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_bars": self.max_drawdown_duration_bars,
            "avg_drawdown": self.avg_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "mar_ratio": self.mar_ratio,
            "return_over_drawdown": self.return_over_drawdown,
            "avg_trade_duration_bars": self.avg_trade_duration_bars,
            "avg_bars_between_trades": self.avg_bars_between_trades,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "longest_losing_streak_days": self.longest_losing_streak_days,
            "r_expectancy": self.r_expectancy,
            "r_multiples": self.r_multiples,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "exit_reason_breakdown": self.exit_reason_breakdown,
            "ulcer_index": self.ulcer_index,
            "ulcer_performance_index": self.ulcer_performance_index,
            "tail_ratio": self.tail_ratio,
            "session_performance": self.session_performance,
            "profitability_score": self.profitability_score,
        }

    def score(self) -> float:
        if self.total_trades < 30:
            return 0.0
        if self.win_rate < 0.40:
            return 0.0
        if self.max_drawdown > 0.20:
            return 0.0

        s = self.sharpe_ratio
        c = self.calmar_ratio
        pf = min(self.profit_factor, 5.0)
        wr = self.win_rate
        re = max(0.0, self.r_expectancy)
        consistency = 1.0 - (self.consecutive_losses / max(self.total_trades, 1))

        raw = (s * 0.30) + (c * 0.25) + (pf * 0.15) + (wr * 0.15) + (re * 0.15)
        return raw * consistency

    def _equity_series(self) -> list[float]:
        if self._equity_curve:
            return list(self._equity_curve)
        equity = self._initial_capital
        series = [equity]
        for trade in self._trades:
            equity += trade.net_pnl
            series.append(equity)
        return series

    def _final_equity(self) -> float:
        if self._equity_curve:
            return float(self._equity_curve[-1])
        return float(self._initial_capital + self.net_profit)
