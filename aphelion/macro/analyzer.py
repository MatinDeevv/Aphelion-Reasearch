"""
MACRO Analyzer — Main coordinator for MACRO intelligence.
Produces regime context for all other modules.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from aphelion.macro.regime import RegimeClassifier, Regime, RegimeState
from aphelion.macro.dxy import DXYMonitor, DXYBias, DXYState
from aphelion.macro.seasonality import GoldSeasonality, SeasonalBias
from aphelion.macro.event_calendar import EconomicCalendar
from aphelion.macro.sentiment import SentimentAnalyzer, SentimentState


@dataclass
class MacroSignal:
    """MACRO output consumed by ARES and all modules."""
    regime: Regime = Regime.RANGING
    regime_confidence: float = 0.5
    dxy_bias: DXYBias = DXYBias.NEUTRAL
    seasonal_bias: str = "NEUTRAL"
    sentiment: str = "NEUTRAL"
    safe_to_trade: bool = True
    event_block_reason: str = ""

    # Sub-states
    regime_state: Optional[RegimeState] = None
    dxy_state: Optional[DXYState] = None
    seasonal_state: Optional[SeasonalBias] = None
    sentiment_state: Optional[SentimentState] = None

    # Derived direction for ARES voting
    direction: int = 0               # 1=BUY, -1=SELL, 0=FLAT
    confidence: float = 0.0          # [0, 1]


class MacroAnalyzer:
    """
    Main MACRO coordinator. Aggregates regime, DXY, seasonality,
    events, and sentiment into a unified MacroSignal.

    Commander-tier ARES voter (10 votes).
    """

    def __init__(self):
        self._regime = RegimeClassifier()
        self._dxy = DXYMonitor()
        self._seasonality = GoldSeasonality()
        self._calendar = EconomicCalendar()
        self._sentiment = SentimentAnalyzer()

    def analyze(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        rsi: float = 50.0,
        current_time=None,
        dxy_price: Optional[float] = None,
        prev_dxy: Optional[float] = None,
    ) -> MacroSignal:
        """Run full MACRO analysis."""
        if len(closes) < 30:
            return MacroSignal()

        # 1. Regime classification
        dxy_trend = "NEUTRAL"
        dxy_state = None
        if dxy_price is not None and prev_dxy is not None:
            gold_price = float(closes[-1])
            prev_gold = float(closes[-2]) if len(closes) >= 2 else gold_price
            dxy_state = self._dxy.update(gold_price, dxy_price, prev_gold, prev_dxy)
            dxy_trend = dxy_state.dxy_trend_1h

        regime_state = self._regime.classify(highs, lows, closes, dxy_trend)

        # 2. Seasonality
        seasonal = self._seasonality.get_bias(current_time)

        # 3. Sentiment
        sentiment = self._sentiment.analyze(closes, volumes, rsi)

        # 4. Event check
        safe = True
        event_reason = ""
        if current_time is not None:
            safe, reason = self._calendar.is_safe_to_trade(current_time)
            event_reason = reason or ""

        # 5. Derive direction for ARES
        direction, confidence = self._compute_direction(
            regime_state, dxy_state, seasonal, sentiment
        )

        # Override to FLAT if not safe to trade
        if not safe:
            direction = 0
            confidence = 0.0

        return MacroSignal(
            regime=regime_state.regime,
            regime_confidence=regime_state.confidence,
            dxy_bias=dxy_state.bias if dxy_state else DXYBias.NEUTRAL,
            seasonal_bias=seasonal.month_bias,
            sentiment=sentiment.sentiment,
            safe_to_trade=safe,
            event_block_reason=event_reason,
            regime_state=regime_state,
            dxy_state=dxy_state,
            seasonal_state=seasonal,
            sentiment_state=sentiment,
            direction=direction,
            confidence=confidence,
        )

    @staticmethod
    def _compute_direction(
        regime: RegimeState,
        dxy: Optional[DXYState],
        seasonal: SeasonalBias,
        sentiment: SentimentState,
    ) -> tuple:
        """Aggregate macro factors into directional bias."""
        score = 0.0

        # Regime contribution
        if regime.regime == Regime.TRENDING_BULL:
            score += 0.3
        elif regime.regime == Regime.TRENDING_BEAR:
            score -= 0.3
        elif regime.regime in (Regime.VOLATILE, Regime.CRISIS):
            return 0, 0.0  # Don't provide directional bias in chaos

        # DXY contribution
        if dxy:
            if dxy.bias == DXYBias.BUY_GOLD:
                score += 0.25
            elif dxy.bias == DXYBias.SELL_GOLD:
                score -= 0.25

        # Seasonal contribution
        if seasonal.month_bias == "BULLISH":
            score += 0.15
        elif seasonal.month_bias == "BEARISH":
            score -= 0.15

        # Sentiment (contrarian at extremes)
        score += sentiment.score * 0.2

        if abs(score) < 0.2:
            return 0, abs(score)
        direction = 1 if score > 0 else -1
        confidence = min(1.0, abs(score))
        return direction, confidence

    @property
    def calendar(self) -> EconomicCalendar:
        return self._calendar

    @property
    def dxy_monitor(self) -> DXYMonitor:
        return self._dxy

    def reset(self) -> None:
        self._dxy.reset()
