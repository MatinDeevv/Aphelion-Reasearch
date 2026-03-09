"""
APHELION PROMETHEUS — Strategy Genome → Backtest Evaluator
Bridges the evolutionary engine to the backtest engine.
Converts a Genome into a strategy callback and runs a full
walk-forward backtest to produce fitness metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aphelion.backtest.engine import BacktestConfig, BacktestEngine
from aphelion.backtest.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, profit_factor,
    max_drawdown, expectancy, win_rate, deflated_sharpe_ratio,
)
from aphelion.backtest.order import Order, OrderType, OrderSide
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import SENTINEL, Session, Timeframe
from aphelion.core.data_layer import Bar
from aphelion.evolution.prometheus.genome import Genome, GenomeFitness

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for the genome evaluator."""
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    num_trials: int = 1                     # For deflated Sharpe calculation
    min_trade_count: int = 10               # Reject if fewer trades


class GenomeStrategy:
    """
    Callable strategy adapter that converts a Genome's gene values
    into trading decisions. Plugs into BacktestEngine.set_strategy().
    """

    def __init__(self, genome: Genome):
        self._genome = genome
        self._cfg = genome.to_strategy_config()
        self._bars_since_trade = 999
        self._trade_counter = 0

    def __call__(
        self,
        bar: Bar,
        features: dict,
        portfolio: Portfolio,
    ) -> list[Order]:
        self._bars_since_trade += 1

        # ── Session filter ───────────────────────────────────────────────
        session = features.get("session", "LONDON")
        session_map = {
            "ASIAN": "allow_asian",
            "LONDON": "allow_london",
            "NEW_YORK": "allow_new_york",
            "OVERLAP_LDN_NY": "allow_overlap",
            "DEAD_ZONE": "allow_dead_zone",
        }
        allowed_key = session_map.get(session, "allow_london")
        if self._cfg.get(allowed_key, 1.0) < 0.5:
            return []

        # ── Cooldown ─────────────────────────────────────────────────────
        cooldown = int(self._cfg.get("cooldown_bars", 5))
        if self._bars_since_trade < cooldown:
            return []

        # ── Max open positions ───────────────────────────────────────────
        max_pos = int(self._cfg.get("max_open_positions", 3))
        if len(portfolio._open_positions) >= max_pos:
            return []

        # ── Compute weighted signal from features ────────────────────────
        signal_score = 0.0
        weight_sum = 0.0

        feature_weights = {
            "vpin": self._cfg.get("weight_vpin", 1.0),
            "ofi": self._cfg.get("weight_ofi", 1.0),
            "vwap_distance": self._cfg.get("weight_vwap_dist", 1.0),
            "atr": self._cfg.get("weight_atr", 1.0),
            "rsi_14": self._cfg.get("weight_rsi", 1.0),
            "spread": self._cfg.get("weight_spread", 1.0),
        }

        for feat_name, weight in feature_weights.items():
            val = features.get(feat_name)
            if val is not None and np.isfinite(val):
                # Normalise features to [-1, 1] range (simplified)
                if feat_name == "rsi_14":
                    normed = (val - 50.0) / 50.0  # RSI: 0-100 → -1 to 1
                elif feat_name == "vwap_distance":
                    normed = np.clip(val / 5.0, -1.0, 1.0)
                elif feat_name == "ofi":
                    normed = np.clip(val / 100.0, -1.0, 1.0)
                elif feat_name == "vpin":
                    normed = val * 2.0 - 1.0  # 0-1 → -1 to 1
                else:
                    normed = np.clip(val / (abs(val) + 1e-10), -1.0, 1.0)
                signal_score += normed * weight
                weight_sum += abs(weight)

        if weight_sum < 1e-10:
            return []

        signal_score /= weight_sum  # Normalise to [-1, 1]

        # ── Confidence threshold check ───────────────────────────────────
        confidence = abs(signal_score)
        threshold = self._cfg.get("confidence_threshold", 0.55)
        if confidence < threshold:
            return []

        # ── Direction ────────────────────────────────────────────────────
        if signal_score > 0:
            side = OrderSide.BUY
        elif signal_score < 0:
            side = OrderSide.SELL
        else:
            return []

        # ── Risk parameters ──────────────────────────────────────────────
        atr = features.get("atr", bar.close * 0.005)
        if atr <= 0:
            atr = bar.close * 0.003

        sl_mult = self._cfg.get("atr_sl_multiplier", 2.0)
        rr = self._cfg.get("rr_ratio", 2.0)
        sl_distance = atr * sl_mult
        tp_distance = sl_distance * rr

        if side == OrderSide.BUY:
            stop_loss = bar.close - sl_distance
            take_profit = bar.close + tp_distance
        else:
            stop_loss = bar.close + sl_distance
            take_profit = bar.close - tp_distance

        # ── Position sizing ──────────────────────────────────────────────
        risk_pct = min(self._cfg.get("risk_per_trade", 0.015), SENTINEL.max_position_pct)

        use_kelly = self._cfg.get("use_kelly", 1.0) >= 0.5
        if use_kelly:
            p = confidence
            q = 1.0 - p
            b = rr
            kelly_raw = (p * b - q) / b if b > 0 else 0.0
            kelly_frac = self._cfg.get("kelly_fraction", 0.25)
            size_pct = min(max(0.0, kelly_raw * kelly_frac), risk_pct)
        else:
            size_pct = risk_pct * confidence

        if size_pct <= 0:
            return []

        equity = portfolio.equity
        risk_dollars = equity * size_pct
        lot_size = risk_dollars / (sl_distance * 100)
        lot_size = max(0.01, round(lot_size, 2))

        self._trade_counter += 1
        order = Order(
            order_id=f"EVO-{self._trade_counter:06d}",
            symbol="XAUUSD",
            order_type=OrderType.MARKET,
            side=side,
            size_lots=lot_size,
            entry_price=0.0,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            size_pct=size_pct,
            proposed_by="PROMETHEUS_NEAT",
        )

        self._bars_since_trade = 0
        return [order]


# ─── Evaluator Function ─────────────────────────────────────────────────────

def evaluate_genome(
    genome: Genome,
    bars: list[Bar],
    engine_factory: callable,
    evaluator_config: Optional[EvaluatorConfig] = None,
) -> GenomeFitness:
    """
    Evaluate a genome by running a backtest and extracting fitness metrics.

    Args:
        genome: The genome to evaluate.
        bars: Historical bar data.
        engine_factory: Callable() -> BacktestEngine (creates a fresh engine).
        evaluator_config: Optional configuration overrides.

    Returns:
        GenomeFitness with all metrics populated.
    """
    cfg = evaluator_config or EvaluatorConfig()

    engine = engine_factory()
    strategy = GenomeStrategy(genome)
    engine.set_strategy(strategy)

    try:
        results = engine.run(bars)
    except Exception:
        logger.warning("Backtest failed for genome %s", genome.genome_id, exc_info=True)
        return GenomeFitness(composite=-999.0)

    trades = results.trades
    daily_rets = results.daily_returns

    fitness = GenomeFitness(
        sharpe=sharpe_ratio(daily_rets) if len(daily_rets) >= 2 else 0.0,
        sortino=sortino_ratio(daily_rets) if len(daily_rets) >= 2 else 0.0,
        calmar=calmar_ratio(daily_rets) if len(daily_rets) >= 2 else 0.0,
        profit_factor=profit_factor(trades) if trades else 0.0,
        total_return_pct=results.total_return_pct,
        max_drawdown=max_drawdown(daily_rets) if daily_rets else 1.0,
        win_rate=win_rate(trades) if trades else 0.0,
        trade_count=len(trades),
        expectancy=expectancy(trades) if trades else 0.0,
    )

    # Deflated Sharpe
    if fitness.trade_count >= cfg.min_trade_count and len(daily_rets) >= 2:
        fitness.dsr = deflated_sharpe_ratio(
            fitness.sharpe, cfg.num_trials, len(daily_rets),
        )

    fitness.compute_composite()
    return fitness
