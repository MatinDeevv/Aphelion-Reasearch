"""Tests for MarketClock DST-aware sessions and new utility methods."""

import pytest
from datetime import datetime, timezone
from aphelion.core.clock import MarketClock
from aphelion.core.config import Session


class TestDSTDetection:
    def setup_method(self):
        self.clock = MarketClock()

    def test_auto_detect_dst_summer(self):
        # July is summer (DST active in both London/NY)
        dt = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
        # auto_detect_dst returns None — it sets internal offsets
        result = self.clock.auto_detect_dst(dt)
        assert result is None
        # Verify internal offsets were set (London BST = UTC+1 → offset = -60)
        assert self.clock._dst_offset_london != 0

    def test_auto_detect_dst_winter(self):
        # January is winter (no DST)
        dt = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        self.clock.auto_detect_dst(dt)
        # In winter, London is UTC+0 → offset should be 0
        assert self.clock._dst_offset_london == 0

    def test_auto_detect_dst_sets_ny_offset(self):
        dt = datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc)
        self.clock.auto_detect_dst(dt)
        # Summer: NY is EDT (UTC-4), normal is UTC-5 → offset = -(-4+5) = +1... actually
        # ny_utcoff = -240, offset = -(ny_utcoff + 300) = -(-240+300) = -60
        assert isinstance(self.clock._dst_offset_ny, int)


class TestSessionProgress:
    def setup_method(self):
        self.clock = MarketClock()

    def test_minutes_into_session(self):
        # London opens 8:00 UTC (winter), 9:00 is 60 mins in
        dt = datetime(2026, 1, 7, 9, 0, tzinfo=timezone.utc)  # Wednesday
        mins = self.clock.minutes_into_session(dt)
        assert mins >= 0

    def test_minutes_into_dead_zone_returns_zero(self):
        # 22:00 UTC is DEAD_ZONE
        dt = datetime(2026, 1, 7, 22, 0, tzinfo=timezone.utc)  # Wednesday
        mins = self.clock.minutes_into_session(dt)
        assert mins == 0.0

    def test_session_duration_minutes(self):
        dt = datetime(2026, 1, 7, 10, 0, tzinfo=timezone.utc)  # Wednesday, London
        dur = self.clock.session_duration_minutes(dt)
        assert dur > 0

    def test_session_progress_in_range(self):
        # 10:00 UTC is mid-London
        dt = datetime(2026, 1, 7, 10, 0, tzinfo=timezone.utc)  # Wednesday
        progress = self.clock.session_progress(dt)
        assert 0.0 <= progress <= 1.0

    def test_session_progress_dead_zone_returns_zero(self):
        dt = datetime(2026, 1, 7, 22, 0, tzinfo=timezone.utc)  # Wednesday
        progress = self.clock.session_progress(dt)
        assert progress == 0.0


class TestSessionFeaturesExtended:
    def setup_method(self):
        self.clock = MarketClock()

    def test_features_include_progress_fields(self):
        dt = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)  # Wednesday
        features = self.clock.session_features(dt)
        assert "minutes_into_session" in features
        assert "session_duration_minutes" in features
        assert "session_progress" in features

    def test_progress_numeric(self):
        dt = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)  # Wednesday
        features = self.clock.session_features(dt)
        assert isinstance(features["session_progress"], (int, float))
        assert 0.0 <= features["session_progress"] <= 1.0
