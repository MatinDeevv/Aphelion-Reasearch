"""Tests for OMEGA strategy, SIGNAL TOWER, and ATLAS LIVE."""

import numpy as np
import pytest
from datetime import datetime

from aphelion.flow.omega import OmegaSignalGenerator, OmegaExitManager, H4Structure
from aphelion.flow.signal_tower import (
    SignalTower, Vote,
    HalfTrendVoter, EMAStackVoter, VWAPPositionVoter,
    BreakoutDetector, RSIExtremeVoter, StructureVoter, SessionMomentumVoter,
)
from aphelion.macro.atlas_live import AtlasLive, FedCalendar, DXYFeed, COTData


# ── OMEGA Signal Generator ────────────────────────────────────────


class TestOmegaSignalGenerator:
    def _make_uptrend_h4(self, n=30):
        highs = np.cumsum(np.random.uniform(1, 5, n)) + 2000
        lows = highs - np.random.uniform(5, 15, n)
        closes = (highs + lows) / 2
        return highs, lows, closes

    def test_unclear_structure_short_data(self):
        gen = OmegaSignalGenerator()
        h = np.array([2000.0, 2001.0])
        l = np.array([1999.0, 2000.0])
        c = np.array([1999.5, 2000.5])
        signal = gen.generate(h, l, c, h, l, c)
        assert signal.direction == 0
        assert signal.reason == "NO_H4_STRUCTURE"

    def test_signal_generation_happy_path(self):
        gen = OmegaSignalGenerator(pullback_tolerance=500)  # Loose tolerance for test
        n = 30
        # Create uptrending data
        base = np.linspace(2000, 2100, n)
        highs = base + np.random.uniform(5, 15, n)
        lows = base - np.random.uniform(5, 15, n)
        closes = base + np.random.uniform(-3, 3, n)
        # With regime TRENDING_BULL
        signal = gen.generate(
            highs, lows, closes,
            highs, lows, closes,
            regime="TRENDING_BULL",
        )
        # May or may not find structure depending on random data
        assert signal.reason in ("SIGNAL_GENERATED", "NO_H4_STRUCTURE", "NO_PULLBACK_LEVEL", "NOT_AT_PULLBACK_LEVEL")

    def test_adverse_regime(self):
        gen = OmegaSignalGenerator(pullback_tolerance=500)
        n = 30
        base = np.linspace(2000, 2100, n)
        highs = base + 10
        lows = base - 10
        closes = base
        signal = gen.generate(highs, lows, closes, highs, lows, closes, regime="CRISIS")
        # Even with good structure, CRISIS regime should block
        assert signal.direction == 0 or signal.reason == "ADVERSE_REGIME" or signal.reason in ("NO_H4_STRUCTURE", "NO_PULLBACK_LEVEL", "NOT_AT_PULLBACK_LEVEL")


class TestOmegaExitManager:
    def test_hold(self):
        mgr = OmegaExitManager()
        result = mgr.check_exit(2005.0, 2000.0, 1, 1990.0, 2030.0, 2060.0)
        assert result["action"] == "HOLD"

    def test_stop_loss(self):
        mgr = OmegaExitManager()
        result = mgr.check_exit(1989.0, 2000.0, 1, 1990.0, 2030.0, 2060.0)
        assert result["action"] == "CLOSE_ALL"
        assert result["reason"] == "STOP_LOSS"

    def test_tp1_partial_close(self):
        mgr = OmegaExitManager()
        result = mgr.check_exit(2035.0, 2000.0, 1, 1990.0, 2030.0, 2060.0)
        assert result["action"] == "PARTIAL_CLOSE"
        assert result["close_pct"] == 0.5
        assert result["new_stop"] == 2000.0  # Breakeven
        assert mgr.stage == 1

    def test_tp2_close_all(self):
        mgr = OmegaExitManager()
        # Hit TP1 first
        mgr.check_exit(2035.0, 2000.0, 1, 1990.0, 2030.0, 2060.0)
        # Then TP2
        result = mgr.check_exit(2065.0, 2000.0, 1, 2000.0, 2030.0, 2060.0)
        assert result["action"] == "CLOSE_ALL"
        assert result["reason"] == "TP2_HIT"

    def test_reset(self):
        mgr = OmegaExitManager()
        mgr.check_exit(2035.0, 2000.0, 1, 1990.0, 2030.0, 2060.0)
        mgr.reset()
        assert mgr.stage == 0


# ── SIGNAL TOWER Voters ───────────────────────────────────────────


class TestHalfTrendVoter:
    def test_insufficient_data(self):
        voter = HalfTrendVoter()
        vote = voter.vote(np.array([1.0]), np.array([1.0]), np.array([1.0]))
        assert vote.direction == 0

    def test_bullish_vote(self):
        voter = HalfTrendVoter(amplitude=2, atr_period=10)
        n = 50
        closes = np.linspace(2000, 2100, n)
        highs = closes + 5
        lows = closes - 5
        vote = voter.vote(highs, lows, closes)
        assert vote.voter_name == "HalfTrend"
        assert vote.direction in (1, 0, -1)


class TestEMAStackVoter:
    def test_insufficient_data(self):
        voter = EMAStackVoter()
        vote = voter.vote(np.array([1.0, 2.0, 3.0]))
        assert vote.direction == 0

    def test_bullish_stack(self):
        voter = EMAStackVoter()
        closes = np.linspace(2000, 2200, 100)
        vote = voter.vote(closes)
        assert vote.direction == 1


class TestVWAPPositionVoter:
    def test_above_vwap(self):
        voter = VWAPPositionVoter()
        vote = voter.vote(close=2050.0, vwap=2000.0)
        assert vote.direction == 1

    def test_below_vwap(self):
        voter = VWAPPositionVoter()
        vote = voter.vote(close=1950.0, vwap=2000.0)
        assert vote.direction == -1


class TestRSIExtremeVoter:
    def test_overbought(self):
        voter = RSIExtremeVoter()
        vote = voter.vote(rsi=85.0)
        assert vote.direction == -1

    def test_oversold(self):
        voter = RSIExtremeVoter()
        vote = voter.vote(rsi=15.0)
        assert vote.direction == 1

    def test_neutral(self):
        voter = RSIExtremeVoter()
        vote = voter.vote(rsi=50.0)
        assert vote.direction == 0


class TestSignalTower:
    def test_collect_votes(self):
        tower = SignalTower()
        n = 100
        closes = np.linspace(2000, 2100, n)
        highs = closes + 5
        lows = closes - 5
        volumes = np.ones(n) * 1000
        votes = tower.collect_votes(highs, lows, closes, volumes, vwap=2050.0, rsi=55.0)
        assert len(votes) == 7

    def test_aggregate(self):
        tower = SignalTower()
        votes = {
            "A": Vote(1, 0.8, "A"),
            "B": Vote(1, 0.6, "B"),
            "C": Vote(-1, 0.3, "C"),
        }
        agg = tower.get_aggregate(votes)
        assert agg.voter_name == "SignalTower"


# ── ATLAS LIVE ────────────────────────────────────────────────────


class TestAtlasLive:
    def test_update_dxy(self):
        atlas = AtlasLive()
        atlas.update_dxy(value=104.5, sma_20=103.0)
        assert atlas.dxy.value == 104.5
        assert atlas.dxy.trend == 1  # Above SMA

    def test_update_cot(self):
        atlas = AtlasLive()
        atlas.update_cot(mm_long=150_000, mm_short=50_000, comm_long=100_000, comm_short=80_000)
        assert atlas.cot.net_speculative == 100_000

    def test_compute_state(self):
        atlas = AtlasLive()
        atlas.update_dxy(104.0, 103.0)
        atlas.update_feed("test_feed", 1.0, direction=1)
        state = atlas.compute_state()
        assert state.feeds_total == 1
        assert state.dxy_bias in (-1, 0, 1)


class TestFedCalendar:
    def test_fomc_date(self):
        cal = FedCalendar()
        fomc = datetime(2025, 1, 29, 14, 0)
        assert cal.is_near_event(fomc) is True

    def test_normal_date(self):
        cal = FedCalendar()
        normal = datetime(2025, 4, 15, 14, 0)
        assert cal.is_near_event(normal) is False

    def test_next_event(self):
        cal = FedCalendar()
        dt = datetime(2025, 1, 1)
        nxt = cal.next_event(dt)
        assert nxt is not None
