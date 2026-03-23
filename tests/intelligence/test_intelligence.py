"""Tests for Intelligence modules — KRONOS, ECHO, FORGE."""

import pytest
import numpy as np
from datetime import datetime

from aphelion.intelligence.kronos.journal import KRONOSJournal, TradeRecord, PerformanceMetrics
from aphelion.intelligence.echo.library import PatternLibrary, PatternEncoder, PatternFingerprint
from aphelion.intelligence.forge.optimizer import ForgeOptimizer, BayesianOptimizer, ParameterSpec


# ── KRONOS Journal ────────────────────────────────────────────────


class TestKRONOSJournal:
    def test_record_entry(self):
        journal = KRONOSJournal()
        trade_id = journal.record_entry(
            direction=1,
            entry_price=2000.0,
            lot_size=0.1,
            stop_loss=1990.0,
            take_profit=2020.0,
            ares_confidence=0.75,
            regime="TRENDING_BULL",
            session="London",
        )
        assert isinstance(trade_id, str)

    def test_record_exit(self):
        journal = KRONOSJournal()
        trade_id = journal.record_entry(
            direction=1, entry_price=2000.0, lot_size=0.1,
            stop_loss=1990.0, take_profit=2020.0,
        )
        result = journal.record_exit(trade_id, exit_price=2015.0, profit_usd=150.0)
        assert result is not None
        assert result.exit_price == 2015.0

    def test_get_trade(self):
        journal = KRONOSJournal()
        tid = journal.record_entry(1, 2000.0, 0.1, 1990.0, 2020.0)
        trade = journal.get_trade(tid)
        assert trade is not None
        assert trade.entry_price == 2000.0

    def test_get_metrics(self):
        journal = KRONOSJournal()
        # Winning trades
        for _ in range(5):
            tid = journal.record_entry(1, 2000.0, 0.1, 1990.0, 2020.0)
            journal.record_exit(tid, 2015.0, 150.0)
        # Losing trades
        for _ in range(3):
            tid = journal.record_entry(1, 2000.0, 0.1, 1990.0, 2020.0)
            journal.record_exit(tid, 1992.0, -80.0)

        metrics = journal.get_metrics()
        assert metrics.total_trades == 8
        assert metrics.win_rate > 0.5

    def test_empty_metrics(self):
        journal = KRONOSJournal()
        metrics = journal.get_metrics()
        assert metrics.total_trades == 0

    def test_trade_count(self):
        journal = KRONOSJournal()
        journal.record_entry(1, 2000.0, 0.1, 1990.0, 2020.0)
        assert journal.trade_count == 1


# ── ECHO Library ──────────────────────────────────────────────────


class TestPatternEncoder:
    def test_encode_features(self):
        encoder = PatternEncoder()
        features = {"rsi": 65.0, "atr": 12.5, "volume": 50000.0}
        vector = encoder.encode(features)
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0


class TestPatternLibrary:
    def test_store_and_count(self):
        library = PatternLibrary()
        fp = PatternFingerprint(
            pattern_id="P1",
            features={"rsi": 65.0, "atr": 12.5},
            direction=1,
            outcome="WIN",
            r_multiple=1.5,
        )
        library.store(fp)
        assert library.pattern_count == 1

    def test_match(self):
        library = PatternLibrary()
        fp = PatternFingerprint(
            pattern_id="P1",
            features={"rsi": 65.0, "atr": 12.5},
            direction=1,
            outcome="WIN",
        )
        library.store(fp)
        matches = library.match({"rsi": 64.0, "atr": 12.0})
        assert isinstance(matches, list)

    def test_confidence_boost(self):
        library = PatternLibrary()
        fp = PatternFingerprint(
            pattern_id="P1",
            features={"rsi": 65.0, "atr": 12.5},
            direction=1,
            outcome="WIN",
        )
        library.store(fp)
        boost = library.get_confidence_boost({"rsi": 64.0, "atr": 12.0})
        assert isinstance(boost, float)

    def test_clear(self):
        library = PatternLibrary()
        fp = PatternFingerprint("P1", {"rsi": 60.0}, 1, "WIN")
        library.store(fp)
        library.clear()
        assert library.pattern_count == 0


# ── FORGE Optimizer ───────────────────────────────────────────────


class TestBayesianOptimizer:
    def test_suggest(self):
        space = [ParameterSpec("test_param", 0.0, 1.0, 0.5, 0.1)]
        opt = BayesianOptimizer(parameter_space=space)
        suggestion = opt.suggest()
        assert "test_param" in suggestion
        assert 0.0 <= suggestion["test_param"] <= 1.0

    def test_report(self):
        space = [ParameterSpec("x", 0.0, 10.0, 5.0, 1.0)]
        opt = BayesianOptimizer(parameter_space=space)
        opt.report(trial_id=1, parameters={"x": 3.0}, fitness=2.0)
        assert opt.trial_count == 1

    def test_best(self):
        space = [ParameterSpec("x", 0.0, 10.0, 5.0, 1.0)]
        opt = BayesianOptimizer(parameter_space=space)
        opt.report(1, {"x": 3.0}, fitness=2.0)
        opt.report(2, {"x": 7.0}, fitness=1.0)
        assert opt.best is not None
        assert opt.best.parameters["x"] == 3.0


class TestForgeOptimizer:
    def test_instantiation(self):
        forge = ForgeOptimizer()
        assert forge is not None

    def test_suggest_and_report(self):
        forge = ForgeOptimizer()
        params = forge.suggest_parameters()
        assert isinstance(params, dict)
        forge.report_result(trial_id=1, params=params, fitness=1.5)
        assert forge.trial_count == 1


# ── SHADOW Generator — REMOVED ───────────────────────────────────
# Synthetic data generation has been removed from Aphelion.
# All tests that used SHADOW have been removed.
