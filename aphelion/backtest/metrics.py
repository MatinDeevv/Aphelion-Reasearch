"""
APHELION Backtest Metrics
Full institutional-grade performance metric suite.
Sharpe, DSR, Calmar, Sortino, Omega, Profit Factor, Expectancy,
R-multiple distribution, monthly P&L, and deployment scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from aphelion.backtest.order import BacktestTrade


# ─── Core metric functions ───────────────────────────────────────────────────


def sharpe_ratio(
    daily_returns: list[float],
    risk_free_annual: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns, dtype=np.float64)
    daily_rf = (1 + risk_free_annual) ** (1 / trading_days) - 1
    excess = arr - daily_rf
    mean_excess = float(np.mean(excess))
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return 0.0
    return mean_excess / std * math.sqrt(trading_days)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    backtest_length_days: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Bailey & López de Prado Deflated Sharpe Ratio.
    Adjusts for multiple testing (trial count), non-normal returns,
    and finite sample length.
    Returns the probability that the observed Sharpe is genuine (0-1).
    """
    if num_trials < 1 or backtest_length_days < 2:
        return 0.0
    from scipy import stats as sp_stats  # local import — optional dep

    # Expected max Sharpe under null (Euler-Mascheroni correction)
    euler = 0.5772156649
    e_max_sr = (
        (1 - euler) * sp_stats.norm.ppf(1 - 1 / num_trials)
        + euler * sp_stats.norm.ppf(1 - 1 / (num_trials * math.e))
    )
    n = backtest_length_days
    # Standard error of Sharpe estimator adjusted for skew/kurtosis
    se = math.sqrt(
        (1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe ** 2)
        / (n - 1)
    )
    if se == 0:
        return 0.0
    t_stat = (observed_sharpe - e_max_sr) / se
    return float(sp_stats.norm.cdf(t_stat))


def sortino_ratio(
    daily_returns: list[float],
    risk_free_annual: float = 0.05,
    trading_days: int = 252,
) -> float:
    """Annualised Sortino ratio (downside risk only)."""
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns, dtype=np.float64)
    daily_rf = (1 + risk_free_annual) ** (1 / trading_days) - 1
    excess = arr - daily_rf
    mean_excess = float(np.mean(excess))
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if mean_excess > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0:
        return 0.0
    return mean_excess / downside_std * math.sqrt(trading_days)


def calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
    years: float = 1.0,
) -> float:
    """Calmar = annualised return / max drawdown."""
    if max_drawdown_pct == 0 or years == 0:
        return 0.0
    annual_return = total_return_pct / years
    return annual_return / max_drawdown_pct


def omega_ratio(
    daily_returns: list[float],
    threshold: float = 0.0,
) -> float:
    """Omega ratio: sum of gains above threshold / sum of losses below."""
    if not daily_returns:
        return 0.0
    arr = np.array(daily_returns, dtype=np.float64)
    gains = np.sum(arr[arr > threshold] - threshold)
    losses = np.sum(threshold - arr[arr <= threshold])
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def profit_factor(trades: list[BacktestTrade]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
    """
    Maximum drawdown from an equity curve.
    Returns (max_dd_pct, peak_index, trough_index).
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    arr = np.array(equity_curve, dtype=np.float64)
    peak = arr[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak = 0
    max_dd_trough = 0
    for i in range(1, len(arr)):
        if arr[i] > peak:
            peak = arr[i]
            peak_idx = i
        dd = (peak - arr[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_peak = peak_idx
            max_dd_trough = i
    return max_dd, max_dd_peak, max_dd_trough


def expectancy(trades: list[BacktestTrade]) -> float:
    """Average net_pnl per trade."""
    if not trades:
        return 0.0
    return sum(t.net_pnl for t in trades) / len(trades)


def win_rate(trades: list[BacktestTrade]) -> float:
    """Fraction of trades with positive net P&L."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.net_pnl > 0)
    return winners / len(trades)


def avg_risk_reward(trades: list[BacktestTrade]) -> float:
    """Average R-multiple across all trades."""
    r_vals = [t.r_multiple for t in trades if t.r_multiple != 0]
    if not r_vals:
        return 0.0
    return sum(r_vals) / len(r_vals)


def avg_win_loss(trades: list[BacktestTrade]) -> tuple[float, float]:
    """Average winning trade and average losing trade (absolute)."""
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    return avg_w, avg_l


def consecutive_stats(trades: list[BacktestTrade]) -> tuple[int, int]:
    """Max consecutive wins and max consecutive losses."""
    max_wins = max_losses = 0
    cur_wins = cur_losses = 0
    for t in trades:
        if t.net_pnl > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        elif t.net_pnl < 0:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
        else:
            cur_wins = cur_losses = 0
    return max_wins, max_losses


def monthly_pnl(trades: list[BacktestTrade]) -> dict[str, float]:
    """Net P&L grouped by month (YYYY-MM)."""
    monthly: dict[str, float] = {}
    for t in trades:
        key = t.exit_time.strftime("%Y-%m") if isinstance(t.exit_time, datetime) else "unknown"
        monthly[key] = monthly.get(key, 0.0) + t.net_pnl
    return monthly


def hourly_pnl(trades: list[BacktestTrade]) -> dict[int, float]:
    """Net P&L grouped by entry hour (0-23 UTC)."""
    hourly: dict[int, float] = {}
    for t in trades:
        hour = t.entry_time.hour if isinstance(t.entry_time, datetime) else 0
        hourly[hour] = hourly.get(hour, 0.0) + t.net_pnl
    return hourly


def exit_reason_breakdown(trades: list[BacktestTrade]) -> dict[str, int]:
    """Count of trades by exit reason."""
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    return reasons


# ─── Full Report ─────────────────────────────────────────────────────────────


@dataclass
class BacktestMetrics:
    """Complete institutional-grade backtest metric report."""

    # Identification
    total_trades: int = 0
    total_bars: int = 0
    backtest_days: int = 0

    # Returns
    total_return_pct: float = 0.0
    final_equity: float = 0.0
    initial_capital: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    omega: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_dd_peak_idx: int = 0
    max_dd_trough_idx: int = 0

    # Trade stats
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy_dollars: float = 0.0
    avg_r_multiple: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    payoff_ratio: float = 0.0
    expectancy_r: float = 0.0
    return_over_max_drawdown: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    profitable_months: int = 0
    losing_months: int = 0
    profitable_month_ratio: float = 0.0

    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    cost_to_gross_profit_pct: float = 0.0

    # Breakdowns
    monthly_pnl: dict = field(default_factory=dict)
    hourly_pnl: dict = field(default_factory=dict)
    exit_reasons: dict = field(default_factory=dict)

    # Profitability composite
    profitability_score: float = 0.0

    # Deployment
    deployment_approved: bool = False
    deployment_reason: str = ""


def compute_metrics(
    trades: list[BacktestTrade],
    equity_curve: list[float],
    daily_returns: list[float],
    initial_capital: float = 10_000.0,
    total_bars: int = 0,
    num_trials: int = 1,
    broker_stats: Optional[dict] = None,
) -> BacktestMetrics:
    """Compute the full institutional metric suite from backtest results."""

    m = BacktestMetrics()
    m.total_trades = len(trades)
    m.total_bars = total_bars
    m.initial_capital = initial_capital
    m.final_equity = equity_curve[-1] if equity_curve else initial_capital
    m.total_return_pct = (
        (m.final_equity - initial_capital) / initial_capital * 100
        if initial_capital > 0
        else 0.0
    )

    # Estimate backtest duration in days
    if trades:
        first = min(t.entry_time for t in trades)
        last = max(t.exit_time for t in trades)
        if isinstance(first, datetime) and isinstance(last, datetime):
            m.backtest_days = max(1, (last - first).days)
    if m.backtest_days == 0:
        m.backtest_days = max(1, total_bars // (24 * 60))  # rough M1 estimate

    years = m.backtest_days / 365.25

    # Risk-adjusted ratios
    m.sharpe = sharpe_ratio(daily_returns)
    m.sortino = sortino_ratio(daily_returns)
    m.calmar = calmar_ratio(m.total_return_pct, 0.0, years)  # placeholder until MDD
    m.omega = omega_ratio(daily_returns)

    # Drawdown
    mdd, peak_i, trough_i = max_drawdown(equity_curve)
    m.max_drawdown_pct = mdd * 100
    m.max_dd_peak_idx = peak_i
    m.max_dd_trough_idx = trough_i
    m.calmar = calmar_ratio(m.total_return_pct, m.max_drawdown_pct, years)

    # DSR (try/except for scipy dependency)
    try:
        skew = float(np.mean(((np.array(daily_returns) - np.mean(daily_returns)) / (np.std(daily_returns) + 1e-12)) ** 3)) if daily_returns else 0.0
        kurt = float(np.mean(((np.array(daily_returns) - np.mean(daily_returns)) / (np.std(daily_returns) + 1e-12)) ** 4)) if daily_returns else 3.0
        m.deflated_sharpe = deflated_sharpe_ratio(
            m.sharpe, num_trials, m.backtest_days, skew, kurt,
        )
    except ImportError:
        m.deflated_sharpe = 0.0  # scipy not installed

    # Trade statistics
    m.win_rate_pct = win_rate(trades) * 100
    m.profit_factor = profit_factor(trades)
    m.expectancy_dollars = expectancy(trades)
    m.avg_r_multiple = avg_risk_reward(trades)
    m.avg_win, m.avg_loss = avg_win_loss(trades)
    m.gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    m.gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    m.net_profit = m.gross_profit - m.gross_loss
    avg_loss_abs = abs(m.avg_loss)
    if avg_loss_abs > 0:
        m.payoff_ratio = m.avg_win / avg_loss_abs if m.avg_win > 0 else 0.0
        m.expectancy_r = m.expectancy_dollars / avg_loss_abs
    else:
        m.payoff_ratio = float("inf") if m.avg_win > 0 else 0.0
        m.expectancy_r = 0.0
    if m.max_drawdown_pct > 0:
        m.return_over_max_drawdown = m.total_return_pct / m.max_drawdown_pct
    else:
        m.return_over_max_drawdown = float("inf") if m.total_return_pct > 0 else 0.0
    m.max_consecutive_wins, m.max_consecutive_losses = consecutive_stats(trades)

    # Costs
    if broker_stats:
        m.total_commission = broker_stats.get("total_commission", 0.0)
        m.total_slippage = broker_stats.get("total_slippage_cost", 0.0)
    else:
        m.total_commission = sum(t.commission for t in trades)

    # Breakdowns
    m.monthly_pnl = monthly_pnl(trades)
    m.hourly_pnl = hourly_pnl(trades)
    m.exit_reasons = exit_reason_breakdown(trades)
    m.profitable_months = sum(1 for pnl in m.monthly_pnl.values() if pnl > 0)
    m.losing_months = sum(1 for pnl in m.monthly_pnl.values() if pnl < 0)
    month_count = len(m.monthly_pnl)
    m.profitable_month_ratio = (
        (m.profitable_months / month_count) * 100 if month_count > 0 else 0.0
    )
    if m.gross_profit > 0:
        m.cost_to_gross_profit_pct = (
            (m.total_commission + m.total_slippage) / m.gross_profit * 100
        )
    else:
        m.cost_to_gross_profit_pct = (
            100.0 if (m.total_commission + m.total_slippage) > 0 else 0.0
        )

    # Profitability-first ranking metric for strategy selection.
    m.profitability_score = _profitability_score(m)

    # Deployment gate (PROMETHEUS fitness criteria)
    m.deployment_approved, m.deployment_reason = _deployment_check(m)

    return m


def _deployment_check(m: BacktestMetrics) -> tuple[bool, str]:
    """
    Check if a backtest result meets minimum deployment thresholds.
    Based on the spec: walk-forward Sharpe > 1.0, 500+ trades,
    NEMESIS Level 5+ certification.
    """
    reasons: list[str] = []

    if m.total_trades < 500:
        reasons.append(f"TOO_FEW_TRADES: {m.total_trades} < 500")
    if m.sharpe < 1.0:
        reasons.append(f"LOW_SHARPE: {m.sharpe:.2f} < 1.0")
    if m.max_drawdown_pct > 15.0:
        reasons.append(f"HIGH_MDD: {m.max_drawdown_pct:.1f}% > 15%")
    if m.win_rate_pct < 45.0:
        reasons.append(f"LOW_WIN_RATE: {m.win_rate_pct:.1f}% < 45%")
    if m.profit_factor < 1.2:
        reasons.append(f"LOW_PF: {m.profit_factor:.2f} < 1.2")
    if m.net_profit <= 0:
        reasons.append(f"NON_POSITIVE_NET_PROFIT: {m.net_profit:.2f} <= 0")
    if m.expectancy_dollars <= 0:
        reasons.append(f"NEG_EXPECTANCY: {m.expectancy_dollars:.2f} <= 0")
    if m.expectancy_r < 0.10:
        reasons.append(f"LOW_EXPECTANCY_R: {m.expectancy_r:.2f} < 0.10")
    if m.payoff_ratio < 1.10:
        reasons.append(f"LOW_PAYOFF: {m.payoff_ratio:.2f} < 1.10")
    if m.return_over_max_drawdown < 1.0:
        reasons.append(
            f"LOW_RETURN_TO_DRAWDOWN: {m.return_over_max_drawdown:.2f} < 1.00"
        )
    if m.profitable_month_ratio < 55.0:
        reasons.append(
            f"LOW_MONTHLY_CONSISTENCY: {m.profitable_month_ratio:.1f}% < 55%"
        )
    if m.cost_to_gross_profit_pct > 35.0:
        reasons.append(
            f"HIGH_COST_LOAD: {m.cost_to_gross_profit_pct:.1f}% > 35%"
        )
    if m.profitability_score < 0.50:
        reasons.append(f"LOW_PROFITABILITY_SCORE: {m.profitability_score:.2f} < 0.50")

    if reasons:
        return False, " | ".join(reasons)
    return True, "APPROVED"


def _profitability_score(m: BacktestMetrics) -> float:
    """
    Profitability-weighted composite score in [0, 1].
    Rewards positive expectancy, efficient payoff, controlled drawdown, and consistency.
    """
    if m.total_trades < 50 or m.net_profit <= 0 or m.expectancy_dollars <= 0:
        return 0.0

    pf_term = min(max(m.profit_factor, 0.0), 3.0) / 3.0
    expectancy_r_term = min(max(m.expectancy_r, 0.0), 1.5) / 1.5
    return_dd_term = min(max(m.return_over_max_drawdown, 0.0), 3.0) / 3.0
    win_term = min(max((m.win_rate_pct - 40.0) / 30.0, 0.0), 1.0)
    month_term = min(max(m.profitable_month_ratio / 100.0, 0.0), 1.0)
    cost_term = 1.0 - min(max(m.cost_to_gross_profit_pct, 0.0), 60.0) / 60.0

    return (
        0.25 * pf_term
        + 0.20 * expectancy_r_term
        + 0.20 * return_dd_term
        + 0.15 * win_term
        + 0.15 * month_term
        + 0.05 * cost_term
    )
