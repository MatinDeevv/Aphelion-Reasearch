"""Tests for APHELION Market Clock."""

from datetime import datetime, timezone, timedelta
from aphelion.core.clock import MarketClock
from aphelion.core.config import Session


class TestMarketClock:
    def setup_method(self):
        self.clock = MarketClock()

    def test_london_session(self):
        # Wednesday 10:00 UTC = London session
        dt = datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc)
        assert self.clock.current_session(dt) == Session.LONDON

    def test_ny_session(self):
        # Wednesday 18:00 UTC = NY session
        dt = datetime(2026, 3, 4, 18, 0, tzinfo=timezone.utc)
        assert self.clock.current_session(dt) == Session.NEW_YORK

    def test_asian_session(self):
        # Wednesday 03:00 UTC = Asian session
        dt = datetime(2026, 3, 4, 3, 0, tzinfo=timezone.utc)
        assert self.clock.current_session(dt) == Session.ASIAN

    def test_overlap_session(self):
        # Wednesday 14:00 UTC = London/NY overlap
        dt = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)
        assert self.clock.current_session(dt) == Session.OVERLAP_LDN_NY

    def test_dead_zone(self):
        # Wednesday 22:00 UTC = Dead zone
        dt = datetime(2026, 3, 4, 22, 0, tzinfo=timezone.utc)
        assert self.clock.current_session(dt) == Session.DEAD_ZONE


class TestMarketOpen:
    def setup_method(self):
        self.clock = MarketClock()

    def test_weekday_open(self):
        dt = datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc)  # Wednesday
        assert self.clock.is_market_open(dt) is True

    def test_saturday_closed(self):
        dt = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)  # Saturday
        assert self.clock.is_market_open(dt) is False

    def test_friday_before_close(self):
        dt = datetime(2026, 3, 6, 20, 0, tzinfo=timezone.utc)  # Friday 20:00
        assert self.clock.is_market_open(dt) is True

    def test_friday_after_close(self):
        dt = datetime(2026, 3, 6, 21, 30, tzinfo=timezone.utc)  # Friday 21:30
        assert self.clock.is_market_open(dt) is False

    def test_sunday_before_open(self):
        dt = datetime(2026, 3, 8, 20, 0, tzinfo=timezone.utc)  # Sunday 20:00
        assert self.clock.is_market_open(dt) is False

    def test_sunday_after_open(self):
        dt = datetime(2026, 3, 8, 23, 0, tzinfo=timezone.utc)  # Sunday 23:00
        assert self.clock.is_market_open(dt) is True


class TestFridayLockout:
    def setup_method(self):
        self.clock = MarketClock()

    def test_friday_lockout_active(self):
        dt = datetime(2026, 3, 6, 20, 35, tzinfo=timezone.utc)  # 25 min before close
        assert self.clock.is_friday_lockout(dt) is True

    def test_friday_no_lockout(self):
        dt = datetime(2026, 3, 6, 18, 0, tzinfo=timezone.utc)  # Well before
        assert self.clock.is_friday_lockout(dt) is False

    def test_wednesday_no_lockout(self):
        dt = datetime(2026, 3, 4, 20, 35, tzinfo=timezone.utc)
        assert self.clock.is_friday_lockout(dt) is False


class TestNewsLockout:
    def setup_method(self):
        self.clock = MarketClock()
        self.clock.set_news_calendar([
            {
                "time": datetime(2026, 3, 6, 13, 30, tzinfo=timezone.utc),
                "impact": "HIGH",
                "name": "NFP",
            },
        ])

    def test_pre_news_lockout(self):
        dt = datetime(2026, 3, 6, 13, 26, tzinfo=timezone.utc)  # 4 min before
        assert self.clock.is_news_lockout(dt) is True

    def test_post_news_lockout(self):
        dt = datetime(2026, 3, 6, 13, 31, tzinfo=timezone.utc)  # 1 min after
        assert self.clock.is_news_lockout(dt) is True

    def test_outside_lockout(self):
        dt = datetime(2026, 3, 6, 13, 0, tzinfo=timezone.utc)  # 30 min before
        assert self.clock.is_news_lockout(dt) is False

    def test_minutes_to_news(self):
        dt = datetime(2026, 3, 6, 13, 0, tzinfo=timezone.utc)
        minutes = self.clock.minutes_to_next_news(dt)
        assert 29.5 <= minutes <= 30.5


class TestSessionFeatures:
    def setup_method(self):
        self.clock = MarketClock()

    def test_features_dict_complete(self):
        dt = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)
        features = self.clock.session_features(dt)
        expected_keys = {
            "session", "minutes_to_london_open", "minutes_to_ny_open",
            "minutes_to_session_close", "day_of_week", "week_of_month",
            "minutes_to_next_news", "last_news_minutes_ago",
            "is_month_end", "is_quarter_end", "is_friday_lockout",
            "is_news_lockout", "market_open", "is_trading_session",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_sin", "dom_cos",
            "minutes_into_session", "session_duration_minutes", "session_progress",
        }
        assert set(features.keys()) == expected_keys

    def test_day_of_week(self):
        dt = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)  # Wednesday
        assert self.clock.day_of_week(dt) == "WED"

    def test_month_end(self):
        dt = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)  # Near month end
        assert self.clock.is_month_end(dt) is True

    def test_not_month_end(self):
        dt = datetime(2026, 3, 15, 14, 0, tzinfo=timezone.utc)
        assert self.clock.is_month_end(dt) is False
