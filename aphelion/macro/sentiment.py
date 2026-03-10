"""
MACRO Sentiment Analyzer
Simple sentiment derived from price action and volume.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SentimentState:
    sentiment: str = "NEUTRAL"   # "BULLISH", "BEARISH", "NEUTRAL", "EXTREME_GREED", "EXTREME_FEAR"
    score: float = 0.0           # [-1, 1] — positive = bullish
    rsi_sentiment: str = "NEUTRAL"
    volume_sentiment: str = "NEUTRAL"
    momentum_sentiment: str = "NEUTRAL"


class SentimentAnalyzer:
    """
    Derives market sentiment from price action indicators.
    Uses RSI extremes, volume trends, and momentum.
    """

    def analyze(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        rsi: float = 50.0,
    ) -> SentimentState:
        """Compute sentiment from recent price data."""
        if len(closes) < 20:
            return SentimentState()

        # RSI-based sentiment
        if rsi > 80:
            rsi_sent = "EXTREME_GREED"
            rsi_score = -0.3  # Contrarian: extreme greed = bearish signal
        elif rsi > 70:
            rsi_sent = "BULLISH"
            rsi_score = -0.1
        elif rsi < 20:
            rsi_sent = "EXTREME_FEAR"
            rsi_score = 0.3   # Contrarian: extreme fear = bullish signal
        elif rsi < 30:
            rsi_sent = "BEARISH"
            rsi_score = 0.1
        else:
            rsi_sent = "NEUTRAL"
            rsi_score = 0.0

        # Volume sentiment: rising volume in direction of trend = confirming
        vol_20 = float(np.mean(volumes[-20:]))
        vol_5 = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else vol_20
        vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0

        price_direction = 1 if closes[-1] > closes[-5] else -1 if len(closes) >= 5 else 0
        if vol_ratio > 1.3 and price_direction > 0:
            vol_sent = "BULLISH"
            vol_score = 0.3
        elif vol_ratio > 1.3 and price_direction < 0:
            vol_sent = "BEARISH"
            vol_score = -0.3
        else:
            vol_sent = "NEUTRAL"
            vol_score = 0.0

        # Momentum sentiment: rate of change
        roc = (closes[-1] - closes[-20]) / closes[-20] * 100
        if roc > 2.0:
            mom_sent = "BULLISH"
            mom_score = 0.4
        elif roc < -2.0:
            mom_sent = "BEARISH"
            mom_score = -0.4
        else:
            mom_sent = "NEUTRAL"
            mom_score = roc / 5  # Scale linearly

        # Aggregate
        total_score = rsi_score * 0.3 + vol_score * 0.3 + mom_score * 0.4
        total_score = max(-1.0, min(1.0, total_score))

        if total_score > 0.4:
            sentiment = "BULLISH"
        elif total_score < -0.4:
            sentiment = "BEARISH"
        elif total_score > 0.7:
            sentiment = "EXTREME_GREED"
        elif total_score < -0.7:
            sentiment = "EXTREME_FEAR"
        else:
            sentiment = "NEUTRAL"

        return SentimentState(
            sentiment=sentiment,
            score=total_score,
            rsi_sentiment=rsi_sent,
            volume_sentiment=vol_sent,
            momentum_sentiment=mom_sent,
        )
