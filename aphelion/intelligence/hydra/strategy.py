"""
APHELION HYDRA Strategy Adapter
Bridges HYDRA inference signals → BacktestEngine strategy callback.
Converts HydraSignal into Order objects for the backtester.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from aphelion.backtest.order import Order, OrderType, OrderSide, OrderStatus
from aphelion.backtest.portfolio import Portfolio
from aphelion.core.config import SENTINEL
from aphelion.core.data_layer import Bar
from aphelion.intelligence.hydra.inference import HydraInference, InferenceConfig
from aphelion.intelligence.hydra.tft import HydraSignal

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


@dataclass
class StrategyConfig:
    """HYDRA strategy configuration."""
    min_confidence: float = 0.55           # Minimum confidence to trade
    min_horizon_agreement: float = 0.66    # At least 2/3 horizons must agree
    risk_per_trade: float = 0.02           # Max 2% risk per trade (SENTINEL limit)
    atr_sl_multiplier: float = 2.0         # SL = entry +/- ATR * multiplier
    rr_ratio: float = 2.0                  # Risk:Reward ratio for TP
    signal_cooldown_bars: int = 5          # Min bars between new trades
    max_open_positions: int = 3            # SENTINEL max simultaneous
    uncertainty_ceiling: float = 0.8       # Reject signals with high uncertainty


class HydraStrategy:
    """
    Backtest strategy adapter for HYDRA TFT.
    Called by BacktestEngine on each bar to produce orders.
    """

    def __init__(
        self,
        inference: HydraInference,
        config: Optional[StrategyConfig] = None,
    ):
        self._inference = inference
        self._config = config or StrategyConfig()
        self._bars_since_trade = 999  # Allow first trade immediately
        self._trade_counter = 0

    def __call__(
        self,
        bar: Bar,
        features: dict,
        portfolio: Portfolio,
    ) -> list[Order]:
        """
        Strategy callback for BacktestEngine.

        Args:
            bar: Current bar data.
            features: Feature dict from FeatureEngine.
            portfolio: Current portfolio state.

        Returns:
            List of orders to submit (usually 0 or 1).
        """
        self._bars_since_trade += 1

        # Run HYDRA inference
        signal = self._inference.update_bar(features, bar.close)

        if signal is None:
            return []

        # ── Gate checks ──────────────────────────────────────────────────

        # Not actionable (FLAT or low confidence)
        if not signal.is_actionable:
            return []

        # Confidence too low
        if signal.confidence < self._config.min_confidence:
            return []

        # Horizons don't agree
        if signal.horizon_agreement < self._config.min_horizon_agreement:
            return []

        # Uncertainty too high
        if signal.uncertainty > self._config.uncertainty_ceiling:
            return []

        # Cooldown not elapsed
        if self._bars_since_trade < self._config.signal_cooldown_bars:
            return []

        # Max positions reached
        open_count = len(portfolio._open_positions)
        if open_count >= self._config.max_open_positions:
            return []

        # ── Build Order ──────────────────────────────────────────────────

        atr = features.get("atr", bar.close * 0.005)  # Fallback: 0.5% of price
        if atr <= 0:
            atr = bar.close * 0.003

        sl_distance = atr * self._config.atr_sl_multiplier
        tp_distance = sl_distance * self._config.rr_ratio

        if signal.direction == "LONG":
            side = OrderSide.BUY
            stop_loss = bar.close - sl_distance
            take_profit = bar.close + tp_distance
        elif signal.direction == "SHORT":
            side = OrderSide.SELL
            stop_loss = bar.close + sl_distance
            take_profit = bar.close - tp_distance
        else:
            return []

        # Compute position size (capped at SENTINEL 2%)
        size_pct = min(self._config.risk_per_trade, SENTINEL.max_position_pct)

        # Scale by confidence: higher confidence → larger position
        confidence_scalar = min(signal.confidence, 1.0)
        size_pct *= confidence_scalar

        # Simple lot size estimate (rough — broker sim handles exact sizing)
        equity = portfolio.equity
        risk_dollars = equity * size_pct
        lot_size = risk_dollars / (sl_distance * 100)  # 100 oz per lot
        lot_size = max(0.01, round(lot_size, 2))       # Min 0.01 lots

        self._trade_counter += 1
        order_id = f"HYDRA-{self._trade_counter:06d}"

        order = Order(
            order_id=order_id,
            symbol="XAUUSD",
            order_type=OrderType.MARKET,
            side=side,
            size_lots=lot_size,
            entry_price=0.0,  # Market order — filled at current price
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            size_pct=size_pct,
            proposed_by="HYDRA_TFT_v1",
        )

        self._bars_since_trade = 0

        return [order]

    def reset(self) -> None:
        """Reset strategy state for new backtest run."""
        self._bars_since_trade = 999
        self._trade_counter = 0
        self._inference.reset()
