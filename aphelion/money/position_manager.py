"""
APHELION Money Makers — Dynamic Position Manager

Implements multiple position sizing strategies:
  - Fixed-Fractional: constant % of equity per trade
  - Kelly Criterion: optimal fraction with quarter-Kelly safety
  - Volatility-Targeted: scale size to target a fixed annualised vol
  - Anti-Martingale: increase size after wins, decrease after losses
  - Optimal-f: fraction that maximises geometric growth (from trade history)

All methods respect SENTINEL hard limits (max_position_pct, lot_size bounds).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from aphelion.core.config import SENTINEL, KELLY_FRACTION, KELLY_MAX_F

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    FIXED_FRACTIONAL = auto()
    KELLY = auto()
    VOLATILITY_TARGET = auto()
    ANTI_MARTINGALE = auto()
    OPTIMAL_F = auto()


@dataclass
class PositionManagerConfig:
    """Configuration for position sizing."""
    method: SizingMethod = SizingMethod.KELLY
    # Fixed fractional
    fixed_risk_pct: float = 0.015           # 1.5% default
    # Kelly
    kelly_fraction: float = KELLY_FRACTION  # Quarter-Kelly
    kelly_max_f: float = KELLY_MAX_F        # Hard cap 2%
    kelly_lookback: int = 50                # Trades for win-rate estimation
    # Volatility target
    target_annual_vol: float = 0.15         # 15% annualised vol target
    vol_lookback_days: int = 20             # Days for realised vol estimate
    # Anti-martingale
    anti_mart_win_bump: float = 1.25        # Increase 25% after win
    anti_mart_loss_cut: float = 0.75        # Decrease 25% after loss
    anti_mart_floor: float = 0.005          # Min 0.5%
    anti_mart_ceiling: float = 0.02         # Max 2%
    # Optimal-f
    optf_lookback: int = 100                # Trades for optimal-f calc
    # Universal
    min_lot_size: float = 0.01
    max_lot_size: float = 10.0
    lot_size_oz: float = 100.0              # 1 lot = 100 oz


@dataclass
class SizeResult:
    """Result of a position sizing calculation."""
    size_pct: float          # Fraction of equity to risk
    size_lots: float         # Lot size
    method_used: SizingMethod
    raw_fraction: float      # Pre-clamp fraction
    confidence: float        # Confidence in the sizing
    notes: str = ""


class PositionManager:
    """
    Dynamic position sizing engine.
    Computes optimal trade size given account state, signal confidence,
    volatility, and recent trade history.
    """

    def __init__(self, config: Optional[PositionManagerConfig] = None):
        self._config = config or PositionManagerConfig()
        self._recent_trades: list[float] = []  # PnL % per trade
        self._current_base_pct: float = self._config.fixed_risk_pct
        self._wins_streak: int = 0
        self._losses_streak: int = 0

    def compute_size(
        self,
        equity: float,
        signal_confidence: float,
        atr: float,
        entry_price: float,
        sl_distance: float,
        daily_returns: Optional[list[float]] = None,
    ) -> SizeResult:
        """
        Compute position size using the configured method.

        Args:
            equity: Current account equity.
            signal_confidence: Strategy confidence [0, 1].
            atr: Current ATR value.
            entry_price: Expected entry price.
            sl_distance: Distance to stop-loss in price units.
            daily_returns: Recent daily returns (for vol-targeting).

        Returns:
            SizeResult with computed lot size and risk fraction.
        """
        method = self._config.method

        if method == SizingMethod.FIXED_FRACTIONAL:
            raw_pct = self._fixed_fractional(signal_confidence)
        elif method == SizingMethod.KELLY:
            raw_pct = self._kelly_sizing(signal_confidence)
        elif method == SizingMethod.VOLATILITY_TARGET:
            raw_pct = self._volatility_target(daily_returns or [])
        elif method == SizingMethod.ANTI_MARTINGALE:
            raw_pct = self._anti_martingale()
        elif method == SizingMethod.OPTIMAL_F:
            raw_pct = self._optimal_f()
        else:
            raw_pct = self._config.fixed_risk_pct

        # Clamp to SENTINEL hard limit
        clamped_pct = min(raw_pct, SENTINEL.max_position_pct)
        clamped_pct = max(clamped_pct, 0.0)

        # Convert to lots
        if sl_distance > 0:
            risk_dollars = equity * clamped_pct
            lots = risk_dollars / (sl_distance * self._config.lot_size_oz)
        else:
            lots = self._config.min_lot_size

        lots = np.clip(lots, self._config.min_lot_size, self._config.max_lot_size)
        lots = round(float(lots), 2)

        return SizeResult(
            size_pct=clamped_pct,
            size_lots=lots,
            method_used=method,
            raw_fraction=raw_pct,
            confidence=signal_confidence,
            notes=f"{method.name} sizing",
        )

    def record_trade(self, pnl_pct: float) -> None:
        """Record a completed trade's PnL % for adaptive sizing."""
        self._recent_trades.append(pnl_pct)
        # Keep bounded
        max_history = max(
            self._config.kelly_lookback,
            self._config.optf_lookback,
            200,
        )
        if len(self._recent_trades) > max_history:
            self._recent_trades = self._recent_trades[-max_history:]
        # Streak tracking
        if pnl_pct > 0:
            self._wins_streak += 1
            self._losses_streak = 0
        elif pnl_pct < 0:
            self._losses_streak += 1
            self._wins_streak = 0

    # ── Sizing Methods ───────────────────────────────────────────────────────

    def _fixed_fractional(self, confidence: float) -> float:
        """Fixed percentage scaled by signal confidence."""
        return self._config.fixed_risk_pct * min(confidence, 1.0)

    def _kelly_sizing(self, confidence: float) -> float:
        """
        Kelly Criterion: f* = (p*b - q) / b
        Where p = win probability, q = 1-p, b = avg_win/avg_loss ratio.
        Uses quarter-Kelly for safety.
        """
        lookback = self._config.kelly_lookback
        trades = self._recent_trades[-lookback:] if self._recent_trades else []

        if len(trades) < 10:
            # Not enough data — use confidence as win-rate proxy
            p = confidence
            b = 2.0  # Assume 2:1 RR
        else:
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            p = len(wins) / len(trades) if trades else 0.5
            avg_win = np.mean(wins) if wins else 0.01
            avg_loss = abs(np.mean(losses)) if losses else 0.01
            b = avg_win / max(avg_loss, 1e-10)

        q = 1.0 - p
        kelly_raw = (p * b - q) / b if b > 0 else 0.0
        kelly_raw = max(0.0, kelly_raw)

        # Apply safety fraction and hard cap
        sized = kelly_raw * self._config.kelly_fraction
        return min(sized, self._config.kelly_max_f)

    def _volatility_target(self, daily_returns: list[float]) -> float:
        """
        Scale position size to target a specific annualised portfolio volatility.
        size_pct = target_vol / (realised_vol * sqrt(252))
        """
        lookback = self._config.vol_lookback_days
        rets = daily_returns[-lookback:] if daily_returns else []

        if len(rets) < 5:
            return self._config.fixed_risk_pct

        realised_vol = float(np.std(rets, ddof=1)) * math.sqrt(252)
        if realised_vol < 1e-10:
            return self._config.fixed_risk_pct

        raw = self._config.target_annual_vol / realised_vol
        return min(raw * self._config.fixed_risk_pct, SENTINEL.max_position_pct)

    def _anti_martingale(self) -> float:
        """
        Anti-martingale: increase after wins, decrease after losses.
        Captures momentum in winning streaks.
        """
        base = self._current_base_pct

        if self._wins_streak > 0:
            factor = self._config.anti_mart_win_bump ** min(self._wins_streak, 5)
            base = self._config.fixed_risk_pct * factor
        elif self._losses_streak > 0:
            factor = self._config.anti_mart_loss_cut ** min(self._losses_streak, 5)
            base = self._config.fixed_risk_pct * factor

        base = np.clip(base, self._config.anti_mart_floor, self._config.anti_mart_ceiling)
        self._current_base_pct = float(base)
        return float(base)

    def _optimal_f(self) -> float:
        """
        Optimal-f: the fraction that maximises the terminal wealth
        given the empirical trade PnL distribution.
        Uses a grid search over [0, max_position_pct].
        """
        lookback = self._config.optf_lookback
        trades = self._recent_trades[-lookback:] if self._recent_trades else []

        if len(trades) < 20:
            return self._config.fixed_risk_pct

        trades_arr = np.array(trades)
        worst_loss = float(np.min(trades_arr))
        if worst_loss >= 0:
            return self._config.fixed_risk_pct

        best_f = self._config.fixed_risk_pct
        best_twrr = -float("inf")

        for f_cand in np.linspace(0.001, SENTINEL.max_position_pct, 50):
            # Terminal wealth relative return
            hpr = 1.0 + f_cand * trades_arr / abs(worst_loss)
            hpr = np.maximum(hpr, 1e-10)
            twrr = float(np.sum(np.log(hpr)))
            if twrr > best_twrr:
                best_twrr = twrr
                best_f = float(f_cand)

        return best_f

    def reset(self) -> None:
        """Reset all adaptive state."""
        self._recent_trades.clear()
        self._current_base_pct = self._config.fixed_risk_pct
        self._wins_streak = 0
        self._losses_streak = 0
