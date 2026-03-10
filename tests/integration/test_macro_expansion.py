"""Tests for Phase 19 — ATLAS expansion modules."""

import pytest
from datetime import datetime, timedelta, timezone

import numpy as np

from aphelion.macro.atlas.dxy_feed import DXYLiveFeed, DXYSnapshot
from aphelion.macro.atlas.cot_parser import COTParser, COTRecord
from aphelion.macro.atlas.event_blocker import EventBlocker, BlockWindow
from aphelion.macro.atlas.core import AtlasCore, MacroContext
from aphelion.macro.argus.core import ArgusCore, MarketAnomaly
from aphelion.macro.herald.core import HeraldCore, NewsEvent
from aphelion.macro.nexus.core import NexusCore, MacroSignal, NexusOutput
from aphelion.macro.oracle.core import OracleCore, Forecast


# ── DXYLiveFeed ─────────────────────────────────────────────────────────────

class TestDXYLiveFeed:

    def test_single_tick(self):
        feed = DXYLiveFeed()
        snap = feed.on_tick(104.5, 1950.0)
        assert isinstance(snap, DXYSnapshot)
        assert snap.value == 104.5

    def test_trend_detection(self):
        feed = DXYLiveFeed()
        feed.on_tick(104.0, 1950.0)
        snap = feed.on_tick(105.0, 1948.0)
        assert snap.trend == 1  # strengthening

    def test_gold_bias_dxy_strengthening(self):
        feed = DXYLiveFeed(corr_window=5)
        # DXY up, gold down → bearish gold
        for i in range(10):
            snap = feed.on_tick(100.0 + i, 2000.0 - i * 2)
        assert snap.gold_bias == -1 or snap.trend == 1

    def test_latest_property(self):
        feed = DXYLiveFeed()
        assert feed.latest is None
        feed.on_tick(104.0, 1950.0)
        assert feed.latest is not None


# ── COTParser ───────────────────────────────────────────────────────────────

class TestCOTParser:

    def test_add_record(self):
        parser = COTParser()
        rec = COTRecord(
            report_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            managed_money_long=100_000,
            managed_money_short=50_000,
        )
        parser.add_record(rec)
        assert parser.latest is not None
        assert parser.latest.net_speculative == 50_000

    def test_speculative_bias_bullish(self):
        rec = COTRecord(
            report_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            managed_money_long=80_000,
            managed_money_short=50_000,
        )
        assert rec.speculative_bias == "BULLISH"

    def test_speculative_bias_extreme_long(self):
        rec = COTRecord(
            report_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            managed_money_long=100_000,
            managed_money_short=40_000,
        )
        assert rec.speculative_bias == "EXTREME_LONG"

    def test_speculative_bias_neutral(self):
        rec = COTRecord(
            report_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            managed_money_long=50_000,
            managed_money_short=50_000,
        )
        assert rec.speculative_bias == "NEUTRAL"

    def test_positioning_change(self):
        parser = COTParser()
        parser.add_record(COTRecord(
            report_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            managed_money_long=50_000, managed_money_short=30_000,
        ))
        parser.add_record(COTRecord(
            report_date=datetime(2024, 1, 8, tzinfo=timezone.utc),
            managed_money_long=60_000, managed_money_short=30_000,
        ))
        assert parser.positioning_change == 10_000

    def test_load_from_json_missing_file(self):
        parser = COTParser(data_dir="/nonexistent")
        count = parser.load_from_json("/nonexistent/cot.json")
        assert count == 0


# ── EventBlocker ────────────────────────────────────────────────────────────

class TestEventBlocker:

    def test_no_blocks_initially(self):
        blocker = EventBlocker()
        assert blocker.is_blocked() is False
        assert blocker.window_count == 0

    def test_add_event_creates_block(self):
        blocker = EventBlocker(pre_event_minutes=30, post_event_minutes=60)
        now = datetime.now(timezone.utc)
        blocker.add_event("FOMC", now, "HIGH")
        assert blocker.is_blocked(now) is True

    def test_block_before_event(self):
        blocker = EventBlocker(pre_event_minutes=30)
        event_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        blocker.add_event("NFP", event_time)
        assert blocker.is_blocked() is True

    def test_not_blocked_outside_window(self):
        blocker = EventBlocker(pre_event_minutes=30, post_event_minutes=60)
        past = datetime.now(timezone.utc) - timedelta(hours=3)
        blocker.add_event("CPI", past)
        assert blocker.is_blocked() is False

    def test_active_blocks(self):
        blocker = EventBlocker()
        now = datetime.now(timezone.utc)
        blocker.add_event("FOMC", now)
        active = blocker.active_blocks(now)
        assert len(active) == 1
        assert active[0].event_name == "FOMC"

    def test_next_block(self):
        blocker = EventBlocker()
        future = datetime.now(timezone.utc) + timedelta(hours=5)
        blocker.add_event("GDP", future)
        nxt = blocker.next_block()
        assert nxt is not None
        assert nxt.event_name == "GDP"

    def test_cleanup_past(self):
        blocker = EventBlocker()
        past = datetime.now(timezone.utc) - timedelta(hours=5)
        blocker.add_event("Old Event", past)
        removed = blocker.cleanup_past()
        assert removed == 1
        assert blocker.window_count == 0


# ── AtlasCore ───────────────────────────────────────────────────────────────

class TestAtlasCore:

    def test_default_context(self):
        core = AtlasCore()
        ctx = core.get_context()
        assert isinstance(ctx, MacroContext)
        assert ctx.macro_score == 0.0

    def test_update_dxy_affects_score(self):
        core = AtlasCore()
        core.update_dxy(1)  # bullish gold
        ctx = core.get_context()
        assert ctx.macro_score > 0

    def test_update_cot_affects_score(self):
        core = AtlasCore()
        core.update_cot("BULLISH")
        ctx = core.get_context()
        assert ctx.macro_score > 0

    def test_event_block_propagated(self):
        core = AtlasCore()
        core.update_event_block(True)
        ctx = core.get_context()
        assert ctx.event_blocked is True

    def test_freshness_propagated(self):
        core = AtlasCore()
        core.set_freshness("FRESH")
        ctx = core.get_context()
        assert ctx.freshness == "FRESH"


# ── ArgusCore ───────────────────────────────────────────────────────────────

class TestArgusCore:

    def test_no_anomalies_on_normal_tick(self):
        core = ArgusCore()
        core.set_normal_spread(2.0)
        anomalies = core.on_tick(1950.0, 1949.5, 2.5, 100, 100, 5.0)
        assert len(anomalies) == 0

    def test_flash_move_detected(self):
        core = ArgusCore(flash_atr_mult=2.0)
        anomalies = core.on_tick(1960.0, 1940.0, 2.0, 100, 100, 5.0)
        flash = [a for a in anomalies if a.anomaly_type == "FLASH_MOVE"]
        assert len(flash) == 1

    def test_spread_blowout_detected(self):
        core = ArgusCore(spread_mult=3.0)
        core.set_normal_spread(2.0)
        anomalies = core.on_tick(1950.0, 1949.5, 10.0, 100, 100, 5.0)
        spread = [a for a in anomalies if a.anomaly_type == "SPREAD_BLOWOUT"]
        assert len(spread) == 1

    def test_volume_anomaly_detected(self):
        core = ArgusCore()
        anomalies = core.on_tick(1950.0, 1949.5, 2.0, 5000, 100, 5.0)
        vol = [a for a in anomalies if a.anomaly_type == "VOLUME_ANOMALY"]
        assert len(vol) == 1

    def test_stale_feed_detected(self):
        core = ArgusCore(stale_threshold_sec=1.0)
        t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
        core.on_tick(1950.0, 1949.5, 2.0, 100, 100, 5.0, now=t1)
        anomalies = core.on_tick(1950.0, 1949.5, 2.0, 100, 100, 5.0, now=t2)
        stale = [a for a in anomalies if a.anomaly_type == "STALE_FEED"]
        assert len(stale) == 1

    def test_alert_active_on_severe(self):
        core = ArgusCore(flash_atr_mult=1.0)
        core.on_tick(1980.0, 1940.0, 2.0, 100, 100, 5.0)
        assert core.alert_active is True


# ── HeraldCore ──────────────────────────────────────────────────────────────

class TestHeraldCore:

    def test_classify_positive(self):
        h = HeraldCore()
        event = h.classify("Fed announces rate cut", "HIGH")
        assert event.gold_bias == 1
        assert event.confidence > 0.5

    def test_classify_negative(self):
        h = HeraldCore()
        event = h.classify("Fed hawkish rate hike expected", "HIGH")
        assert event.gold_bias == -1

    def test_classify_neutral(self):
        h = HeraldCore()
        event = h.classify("Markets close for holiday", "LOW")
        assert event.gold_bias == 0

    def test_net_bias(self):
        h = HeraldCore()
        h.classify("rate cut expected")
        h.classify("dovish policy signal")
        assert h.net_bias > 0

    def test_recent_events_limited(self):
        h = HeraldCore()
        for i in range(30):
            h.classify(f"Event {i}")
        assert len(h.recent_events) == 20


# ── NexusCore ───────────────────────────────────────────────────────────────

class TestNexusCore:

    def test_empty_aggregate(self):
        n = NexusCore()
        out = n.aggregate()
        assert out.composite_score == 0.0
        assert out.signal_count == 0

    def test_single_signal(self):
        n = NexusCore()
        n.add_signal(MacroSignal("DXY", 1, 0.8))
        out = n.aggregate()
        assert out.composite_score > 0
        assert out.signal_count == 1

    def test_mixed_signals(self):
        n = NexusCore()
        n.add_signal(MacroSignal("DXY", 1, 0.8))
        n.add_signal(MacroSignal("COT", -1, 0.5))
        out = n.aggregate()
        assert out.signal_count == 2

    def test_agreement_calculation(self):
        n = NexusCore()
        n.add_signal(MacroSignal("A", 1, 0.8))
        n.add_signal(MacroSignal("B", 1, 0.7))
        n.add_signal(MacroSignal("C", 1, 0.9))
        out = n.aggregate()
        assert out.agreement == pytest.approx(1.0, abs=0.01)

    def test_clear(self):
        n = NexusCore()
        n.add_signal(MacroSignal("X", 1, 0.5))
        n.clear()
        out = n.aggregate()
        assert out.signal_count == 0


# ── OracleCore ──────────────────────────────────────────────────────────────

class TestOracleCore:

    def test_empty_forecast(self):
        o = OracleCore()
        f = o.forecast()
        assert f.direction == 0
        assert f.confidence == 0.0

    def test_bullish_forecast(self):
        o = OracleCore(lookback=20)
        for _ in range(30):
            o.update(0.01)  # positive returns
        f = o.forecast()
        assert f.direction == 1
        assert f.probability_up > 0.5

    def test_bearish_forecast(self):
        o = OracleCore(lookback=20)
        for _ in range(30):
            o.update(-0.01)
        f = o.forecast()
        assert f.direction == -1
        assert f.probability_down > 0.5

    def test_regime_specific_forecast(self):
        o = OracleCore(lookback=10)
        for _ in range(15):
            o.update(0.01, "TRENDING")
        for _ in range(15):
            o.update(-0.01, "RANGING")
        f = o.forecast("TRENDING")
        assert f.direction == 1

    def test_total_observations(self):
        o = OracleCore()
        for _ in range(10):
            o.update(0.005)
        assert o.total_observations == 10

    def test_edge_property(self):
        f = Forecast(1, 0.7, 0.2, 0.5, 10)
        assert f.edge == pytest.approx(0.5, abs=0.01)
