"""Tests for APHELION config validation improvements."""

import pytest
from aphelion.core.config import (
    SentinelLimits,
    Session,
    Timeframe,
    TIMEFRAMES,
    TIMEFRAME_SECONDS,
    SessionWindow,
)


class TestSentinelLimitsValidation:
    """Validate SentinelLimits __post_init__ enforcement."""

    def test_default_limits_valid(self):
        limits = SentinelLimits()
        assert limits.daily_equity_drawdown_l1 < limits.daily_equity_drawdown_l2 < limits.daily_equity_drawdown_l3

    def test_l1_must_be_less_than_l2(self):
        with pytest.raises(ValueError, match="Breaker tiers"):
            SentinelLimits(daily_equity_drawdown_l1=0.10, daily_equity_drawdown_l2=0.05, daily_equity_drawdown_l3=0.15)

    def test_l2_must_be_less_than_l3(self):
        with pytest.raises(ValueError, match="Breaker tiers"):
            SentinelLimits(daily_equity_drawdown_l1=0.03, daily_equity_drawdown_l2=0.12, daily_equity_drawdown_l3=0.08)

    def test_l3_must_be_at_most_one(self):
        with pytest.raises(ValueError, match="Breaker tiers"):
            SentinelLimits(daily_equity_drawdown_l1=0.03, daily_equity_drawdown_l2=0.06, daily_equity_drawdown_l3=1.5)

    def test_zero_l1_rejected(self):
        with pytest.raises(ValueError, match="Breaker tiers"):
            SentinelLimits(daily_equity_drawdown_l1=0.0, daily_equity_drawdown_l2=0.06, daily_equity_drawdown_l3=0.10)

    def test_negative_l1_rejected(self):
        with pytest.raises(ValueError, match="Breaker tiers"):
            SentinelLimits(daily_equity_drawdown_l1=-0.03, daily_equity_drawdown_l2=0.06, daily_equity_drawdown_l3=0.10)

    def test_max_position_pct_positive(self):
        with pytest.raises(ValueError, match="max_position_pct"):
            SentinelLimits(max_position_pct=0.0)

    def test_min_risk_reward_positive(self):
        with pytest.raises(ValueError, match="min_risk_reward"):
            SentinelLimits(min_risk_reward=0.0)

    def test_max_simultaneous_positions_at_least_one(self):
        with pytest.raises(ValueError, match="max_simultaneous"):
            SentinelLimits(max_simultaneous_positions=0)

    def test_valid_custom_limits(self):
        limits = SentinelLimits(daily_equity_drawdown_l1=0.01, daily_equity_drawdown_l2=0.05, daily_equity_drawdown_l3=0.10)
        assert limits.daily_equity_drawdown_l1 == 0.01
        assert limits.daily_equity_drawdown_l2 == 0.05
        assert limits.daily_equity_drawdown_l3 == 0.10


class TestTimeframeExpansion:
    """D1 and W1 timeframe additions."""

    def test_d1_exists(self):
        assert Timeframe.D1.value == "1d"

    def test_w1_exists(self):
        assert Timeframe.W1.value == "1w"

    def test_timeframes_operational_list(self):
        assert Timeframe.M1 in TIMEFRAMES
        assert Timeframe.M5 in TIMEFRAMES
        assert Timeframe.M15 in TIMEFRAMES
        assert Timeframe.H1 in TIMEFRAMES
        # D1/W1 are NOT in operational default list
        assert Timeframe.D1 not in TIMEFRAMES
        assert Timeframe.W1 not in TIMEFRAMES

    def test_timeframe_seconds_complete(self):
        for tf in Timeframe:
            assert tf in TIMEFRAME_SECONDS, f"{tf} missing from TIMEFRAME_SECONDS"

    def test_timeframe_seconds_d1(self):
        assert TIMEFRAME_SECONDS[Timeframe.D1] == 86400

    def test_timeframe_seconds_w1(self):
        assert TIMEFRAME_SECONDS[Timeframe.W1] == 604800


class TestSessionWindow:
    def test_adjusted_shifts_open(self):
        window = SessionWindow(name=Session.LONDON, open_hour=8, open_minute=0, close_hour=16, close_minute=0)
        shifted = window.adjusted(60)
        assert shifted.open_hour == 9
        assert shifted.open_minute == 0
        assert shifted.close_hour == 17
        assert shifted.close_minute == 0

    def test_adjusted_negative(self):
        window = SessionWindow(name=Session.LONDON, open_hour=10, open_minute=0, close_hour=18, close_minute=0)
        shifted = window.adjusted(-30)
        assert shifted.open_hour == 9
        assert shifted.open_minute == 30
        assert shifted.close_hour == 17
        assert shifted.close_minute == 30

    def test_adjusted_preserves_name(self):
        window = SessionWindow(name=Session.LONDON, open_hour=8, open_minute=0, close_hour=16, close_minute=30)
        shifted = window.adjusted(0)
        assert shifted.name == Session.LONDON
