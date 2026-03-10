"""
MACRO Regime Classifier
Classifies the current market into one of five regimes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Regime(Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


@dataclass
class RegimeState:
    regime: Regime = Regime.RANGING
    adx: float = 0.0
    atr: float = 0.0
    atr_percentile: float = 0.5
    confidence: float = 0.5
    dxy_trend: str = "NEUTRAL"


class RegimeClassifier:
    """
    Classifies XAU/USD market regime from price action and optional DXY data.

    Regimes:
    - TRENDING_BULL: Gold uptrend, DXY weakening
    - TRENDING_BEAR: Gold downtrend, DXY strengthening
    - RANGING: Sideways, compression, low volatility
    - VOLATILE: High ATR, news-driven, unpredictable
    - CRISIS: Extreme volatility, black swan territory
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14,
                 atr_lookback: int = 100):
        self._adx_period = adx_period
        self._atr_period = atr_period
        self._atr_lookback = atr_lookback

    def classify(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        dxy_trend: str = "NEUTRAL",
    ) -> RegimeState:
        """Classify current regime from OHLC data."""
        if len(closes) < self._adx_period * 2:
            return RegimeState()

        adx = self._compute_adx(highs, lows, closes, self._adx_period)
        atr = self._compute_atr(highs, lows, closes, self._atr_period)
        atr_pct = self._get_atr_percentile(highs, lows, closes)

        # Classification logic
        if atr_pct > 0.95:
            regime = Regime.CRISIS
            confidence = min(1.0, atr_pct)
        elif atr_pct > 0.80:
            regime = Regime.VOLATILE
            confidence = 0.7 + (atr_pct - 0.80) * 1.5
        elif adx > 25 and dxy_trend == "DOWN":
            regime = Regime.TRENDING_BULL
            confidence = min(1.0, adx / 50)
        elif adx > 25 and dxy_trend == "UP":
            regime = Regime.TRENDING_BEAR
            confidence = min(1.0, adx / 50)
        elif adx > 25:
            # Strong trend but unknown DXY — infer from price direction
            price_direction = closes[-1] - closes[-20] if len(closes) >= 20 else 0
            if price_direction > 0:
                regime = Regime.TRENDING_BULL
            else:
                regime = Regime.TRENDING_BEAR
            confidence = min(1.0, adx / 50) * 0.8  # Lower confidence without DXY
        else:
            regime = Regime.RANGING
            confidence = 1.0 - adx / 25

        return RegimeState(
            regime=regime,
            adx=adx,
            atr=atr,
            atr_percentile=atr_pct,
            confidence=min(1.0, confidence),
            dxy_trend=dxy_trend,
        )

    def _get_atr_percentile(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> float:
        """Compute current ATR's percentile in recent history."""
        lookback = min(self._atr_lookback, len(closes) - 1)
        if lookback < 10:
            return 0.5

        current_atr = self._compute_atr(highs, lows, closes, self._atr_period)
        atr_history = []
        for i in range(self._atr_period + 1, lookback):
            hist_atr = self._compute_atr(
                highs[-i - self._atr_period:-i] if i > 0 else highs[-self._atr_period:],
                lows[-i - self._atr_period:-i] if i > 0 else lows[-self._atr_period:],
                closes[-i - self._atr_period:-i] if i > 0 else closes[-self._atr_period:],
                self._atr_period,
            )
            atr_history.append(hist_atr)

        if not atr_history:
            return 0.5

        below = sum(1 for a in atr_history if a < current_atr)
        return below / len(atr_history)

    @staticmethod
    def _compute_atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
    ) -> float:
        if len(closes) < 2:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _compute_adx(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
    ) -> float:
        n = len(highs)
        if n < 2 * period:
            return 0.0

        up_moves = highs[1:] - highs[:-1]
        down_moves = lows[:-1] - lows[1:]
        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)
        tr_vals = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        atr_sum = np.sum(tr_vals[:period])
        pdm_sum = np.sum(plus_dm[:period])
        mdm_sum = np.sum(minus_dm[:period])
        dx_list = []

        for i in range(period, len(tr_vals)):
            atr_sum = atr_sum - atr_sum / period + tr_vals[i]
            pdm_sum = pdm_sum - pdm_sum / period + plus_dm[i]
            mdm_sum = mdm_sum - mdm_sum / period + minus_dm[i]
            if atr_sum == 0:
                dx_list.append(0.0)
                continue
            plus_di = 100.0 * pdm_sum / atr_sum
            minus_di = 100.0 * mdm_sum / atr_sum
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_list.append(0.0)
            else:
                dx_list.append(100.0 * abs(plus_di - minus_di) / di_sum)

        if not dx_list:
            return 0.0
        adx = np.mean(dx_list[:period]) if len(dx_list) >= period else np.mean(dx_list)
        for i in range(period, len(dx_list)):
            adx = (adx * (period - 1) + dx_list[i]) / period
        return float(adx)
