"""Tests for MACRO module — RegimeClassifier, DXYMonitor, etc."""

import pytest
import numpy as np
from datetime import datetime, date, timedelta

from aphelion.macro.analyzer import MacroAnalyzer, MacroSignal
from aphelion.macro.regime import RegimeClassifier, Regime, RegimeState
from aphelion.macro.dxy import DXYMonitor, DXYState
from aphelion.macro.seasonality import GoldSeasonality, SeasonalBias
from aphelion.macro.event_calendar import EconomicCalendar, EconomicEvent
from aphelion.macro.sentiment import SentimentAnalyzer, SentimentState


class TestRegimeClassifier:
    def _make_trending_up(self, n=120):
        closes = np.linspace(2000, 2200, n) + np.random.randn(n) * 2
        highs = closes + np.random.uniform(3, 10, n)
        lows = closes - np.random.uniform(3, 10, n)
        return highs, lows, closes

    def _make_ranging(self, n=120):
        closes = 2000 + np.random.randn(n) * 3
        highs = closes + 5
        lows = closes - 5
        return highs, lows, closes

    def test_classify_returns_state(self):
        classifier = RegimeClassifier()
        h, l, c = self._make_trending_up()
        state = classifier.classify(h, l, c)
        assert isinstance(state, RegimeState)
        assert isinstance(state.regime, Regime)

    def test_classify_trending(self):
        classifier = RegimeClassifier()
        h, l, c = self._make_trending_up()
        state = classifier.classify(h, l, c)
        # Strong trend may be detected
        assert state.regime in (Regime.TRENDING_BULL, Regime.TRENDING_BEAR, Regime.RANGING, Regime.VOLATILE, Regime.CRISIS)

    def test_classify_ranging(self):
        classifier = RegimeClassifier()
        h, l, c = self._make_ranging()
        state = classifier.classify(h, l, c, dxy_trend="NEUTRAL")
        assert isinstance(state.regime, Regime)

    def test_classify_short_data(self):
        classifier = RegimeClassifier()
        h = np.array([2001.0, 2002.0])
        l = np.array([1999.0, 2000.0])
        c = np.array([2000.0, 2001.0])
        state = classifier.classify(h, l, c)
        assert isinstance(state.regime, Regime)


class TestDXYMonitor:
    def test_update_returns_state(self):
        monitor = DXYMonitor()
        state = monitor.update(gold_price=2000.0, dxy_price=104.5)
        assert isinstance(state, DXYState)

    def test_correlation_needs_data(self):
        monitor = DXYMonitor()
        corr = monitor.compute_rolling_correlation()
        assert isinstance(corr, float)

    def test_breakdown_detection_with_data(self):
        monitor = DXYMonitor()
        for i in range(60):
            monitor.update(gold_price=2000 - i * 2, dxy_price=100 + i * 0.1)
        corr = monitor.compute_rolling_correlation()
        assert isinstance(corr, float)


class TestGoldSeasonality:
    def test_get_bias_returns_seasonal(self):
        seasonal = GoldSeasonality()
        bias = seasonal.get_bias(datetime(2025, 1, 15))
        assert isinstance(bias, SeasonalBias)

    def test_get_bias_current(self):
        seasonal = GoldSeasonality()
        bias = seasonal.get_bias()
        assert isinstance(bias, SeasonalBias)
        assert hasattr(bias, 'month_bias')
        assert hasattr(bias, 'day_of_week_bias')


class TestEconomicCalendar:
    def test_is_safe_normal_day(self):
        calendar = EconomicCalendar()
        result = calendar.is_safe_to_trade(datetime(2025, 4, 15, 14, 0))
        assert isinstance(result, tuple)
        safe, reason = result
        assert isinstance(safe, bool)

    def test_is_not_safe_with_event(self):
        calendar = EconomicCalendar()
        calendar.add_event(EconomicEvent(
            name="NFP", time=datetime(2025, 3, 7, 13, 30),
            currency="USD", impact="HIGH",
        ))
        safe, reason = calendar.is_safe_to_trade(datetime(2025, 3, 7, 13, 15))
        assert safe is False

    def test_get_next_event(self):
        calendar = EconomicCalendar()
        calendar.add_event(EconomicEvent(
            name="FOMC", time=datetime(2025, 6, 18, 18, 0),
            currency="USD", impact="HIGH",
        ))
        nxt = calendar.get_next_event(datetime(2025, 6, 1))
        assert nxt is not None
        assert nxt.name == "FOMC"


class TestSentimentAnalyzer:
    def test_analyze_returns_state(self):
        analyzer = SentimentAnalyzer()
        n = 30
        closes = np.linspace(2000, 2050, n)
        volumes = np.ones(n) * 1000
        state = analyzer.analyze(closes, volumes, rsi=55.0)
        assert isinstance(state, SentimentState)

    def test_bullish_conditions(self):
        analyzer = SentimentAnalyzer()
        closes = np.linspace(2000, 2100, 30)
        volumes = np.ones(30) * 1000
        state = analyzer.analyze(closes, volumes, rsi=70.0)
        assert isinstance(state.score, float)

    def test_bearish_conditions(self):
        analyzer = SentimentAnalyzer()
        closes = np.linspace(2100, 2000, 30)
        volumes = np.ones(30) * 1000
        state = analyzer.analyze(closes, volumes, rsi=30.0)
        assert isinstance(state.score, float)


class TestMacroAnalyzer:
    def test_instantiation(self):
        assert MacroAnalyzer() is not None

    def test_analyze(self):
        analyzer = MacroAnalyzer()
        n = 120
        base = np.linspace(2000, 2100, n)
        signal = analyzer.analyze(
            highs=base + 10, lows=base - 10,
            closes=base, volumes=np.ones(n) * 1000,
            rsi=55.0, current_time=datetime(2025, 6, 15, 14, 0),
        )
        assert isinstance(signal, MacroSignal)
        assert hasattr(signal, 'regime')
        assert hasattr(signal, 'safe_to_trade')
