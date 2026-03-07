"""
APHELION Feature Engine
Master pipeline orchestrating all 60+ feature computations.
"""

import numpy as np
import pandas as pd
from typing import Optional

from aphelion.core.config import Timeframe, TIMEFRAMES
from aphelion.core.clock import MarketClock
from aphelion.core.data_layer import DataLayer, Tick, Bar
from aphelion.features.microstructure import MicrostructureEngine
from aphelion.features.market_structure import MarketStructureEngine
from aphelion.features.volume_profile import VolumeProfileEngine
from aphelion.features.vwap import VWAPCalculator
from aphelion.features.sessions import SessionFeatures
from aphelion.features.mtf import MTFAlignmentEngine
from aphelion.features.cointegration import CointegrationEngine


class FeatureEngine:
    """
    Master feature pipeline. Orchestrates all sub-engines and produces
    a unified feature dict on every bar update.
    """

    def __init__(self, data_layer: DataLayer, clock: Optional[MarketClock] = None):
        self._data = data_layer
        self._clock = clock or MarketClock()

        # Sub-engines
        self._micro = MicrostructureEngine()
        self._structure = MarketStructureEngine()
        self._volume = VolumeProfileEngine()
        self._vwap = {tf: VWAPCalculator() for tf in TIMEFRAMES}
        self._sessions = SessionFeatures(self._clock)
        self._mtf = MTFAlignmentEngine()
        self._cointegration = CointegrationEngine()

        # Technical indicator caches
        self._atr_cache: dict[Timeframe, float] = {}
        self._bb_cache: dict[Timeframe, dict] = {}

        # External data for cointegration (populated by ATLAS/DATA)
        self._external_prices: dict[str, np.ndarray] = {}

    def on_tick(self, tick: Tick) -> dict:
        """Process a raw tick. Returns microstructure features."""
        return self._micro.update(
            timestamp=tick.timestamp,
            bid=tick.bid,
            ask=tick.ask,
            last_price=tick.last,
            volume=tick.volume,
        ).vpin  # Return just VPIN for quick access; full features on bar

    def on_bar(self, bar: Bar) -> dict:
        """
        Process a completed bar. Computes all features for this timeframe.
        Returns the full feature dictionary.
        """
        tf = bar.timeframe
        df = self._data.get_bars_df(tf, count=500)

        if df.empty:
            return {}

        features = {}

        # 1. Microstructure (already updated per-tick, grab latest state)
        features.update(self._micro.to_dict())

        # 2. Market structure
        if len(df) >= 20:
            structure_features = self._structure.compute_all(df)
            features.update(structure_features)

        # 3. Volume profile
        self._volume.update_bar(bar.open, bar.high, bar.low, bar.close, bar.volume)
        self._volume.compute_session_profile(df)
        features.update(self._volume.to_dict())

        # 4. VWAP
        self._vwap[tf].update(bar.high, bar.low, bar.close, bar.volume)
        features.update(self._vwap[tf].to_dict())

        # 5. Technical indicators
        features.update(self._compute_technicals(df, tf))

        # 6. Session features
        features.update(self._sessions.compute())

        # 7. MTF alignment (only on M5+ bars to avoid noise)
        if tf != Timeframe.M1:
            bars_by_tf = {}
            for t in TIMEFRAMES:
                t_df = self._data.get_bars_df(t, count=100)
                if not t_df.empty:
                    bars_by_tf[t] = t_df
            features.update(self._mtf.compute(bars_by_tf))

        # 8. Cointegration (on H1 bars only — too expensive for M1)
        if tf == Timeframe.H1 and self._external_prices:
            xau_prices = df["close"].values
            self._external_prices["XAUUSD"] = xau_prices
            coint_features = self._cointegration.compute_all(self._external_prices)
            features["any_cointegrated"] = coint_features["any_cointegrated"]
            features["max_spread_zscore"] = coint_features["max_spread_zscore"]

        # Add metadata
        features["timeframe"] = tf.value
        features["bar_timestamp"] = str(bar.timestamp)

        return features

    def set_external_prices(self, symbol: str, prices: np.ndarray) -> None:
        """Set external asset prices for cointegration analysis."""
        self._external_prices[symbol] = prices

    def set_mtf_weights(self, weights: dict[Timeframe, float]) -> None:
        """Set dynamic MTF weights from MERIDIAN."""
        self._mtf.set_weights(weights)

    def reset_session(self) -> None:
        """Reset session-level accumulators (called at session open)."""
        for vwap in self._vwap.values():
            vwap.reset_session()
        self._volume.reset_session()

    def _compute_technicals(self, df: pd.DataFrame, tf: Timeframe) -> dict:
        """Compute standard technical indicators."""
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        features = {}

        # ATR (14-period)
        if len(df) >= 15:
            atr = self._compute_atr(highs, lows, closes, period=14)
            self._atr_cache[tf] = atr
            features["atr"] = atr

        # Bollinger Bands (20-period, 2 std)
        if len(closes) >= 20:
            bb = self._compute_bollinger(closes, period=20, std_mult=2.0)
            self._bb_cache[tf] = bb
            features.update(bb)

        # RSI (14-period)
        if len(closes) >= 15:
            features["rsi"] = self._compute_rsi(closes, period=14)

        # EMA 20, 50
        if len(closes) >= 50:
            features["ema_20"] = self._ema(closes, 20)
            features["ema_50"] = self._ema(closes, 50)
            features["ema_cross"] = 1 if features["ema_20"] > features["ema_50"] else -1

        return features

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
        """Average True Range."""
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _compute_bollinger(closes: np.ndarray, period: int = 20,
                           std_mult: float = 2.0) -> dict:
        """Bollinger Bands."""
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        width = (upper - lower) / sma if sma > 0 else 0

        return {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_width": width,
            "bb_percentile": (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5,
        }

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Exponential Moving Average (last value)."""
        multiplier = 2.0 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)
