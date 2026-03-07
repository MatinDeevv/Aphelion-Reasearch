"""Tests for APHELION configuration."""

from aphelion.core.config import (
    SENTINEL, Session, Timeframe, Tier, TIER_VOTE_WEIGHTS,
    MODULES, LEVERAGE_TIERS, GAUNTLET_LEVELS, KELLY_FRACTION,
    SESSION_WINDOWS,
)


class TestSentinelLimits:
    def test_max_position_pct(self):
        assert SENTINEL.max_position_pct == 0.02

    def test_max_simultaneous_positions(self):
        assert SENTINEL.max_simultaneous_positions == 3

    def test_stop_loss_mandatory(self):
        assert SENTINEL.stop_loss_mandatory is True

    def test_min_risk_reward(self):
        assert SENTINEL.min_risk_reward == 1.5

    def test_pre_news_lockout(self):
        assert SENTINEL.pre_news_lockout_minutes == 5

    def test_post_news_lockout(self):
        assert SENTINEL.post_news_lockout_minutes == 2

    def test_friday_lockout(self):
        assert SENTINEL.friday_close_lockout_minutes == 30

    def test_drawdown_l3(self):
        assert SENTINEL.daily_equity_drawdown_l3 == 0.10

    def test_frozen(self):
        """SENTINEL limits must be immutable."""
        try:
            SENTINEL.max_position_pct = 0.05
            assert False, "Should not be able to modify SENTINEL"
        except (AttributeError, TypeError):
            pass


class TestTierVoting:
    def test_sovereign_infinite(self):
        assert TIER_VOTE_WEIGHTS[Tier.SOVEREIGN] == float('inf')

    def test_council_100(self):
        assert TIER_VOTE_WEIGHTS[Tier.COUNCIL] == 100

    def test_minister_40(self):
        assert TIER_VOTE_WEIGHTS[Tier.MINISTER] == 40

    def test_commander_10(self):
        assert TIER_VOTE_WEIGHTS[Tier.COMMANDER] == 10

    def test_operator_1(self):
        assert TIER_VOTE_WEIGHTS[Tier.OPERATOR] == 1


class TestModules:
    def test_sentinel_is_council(self):
        assert MODULES["SENTINEL"].tier == Tier.COUNCIL

    def test_hydra_is_minister(self):
        assert MODULES["HYDRA"].tier == Tier.MINISTER

    def test_venom_is_commander(self):
        assert MODULES["VENOM"].tier == Tier.COMMANDER

    def test_all_modules_present(self):
        expected = {
            "OLYMPUS", "SENTINEL", "ARES", "HYDRA", "PROMETHEUS",
            "PHANTOM", "NEMESIS", "FORGE", "ATLAS", "DATA",
            "VENOM", "REAPER", "APEX", "WRAITH", "SHADOW",
            "KRONOS", "ECHO", "CASSANDRA", "ORACLE", "TITAN",
            "GHOST", "FUND",
        }
        assert set(MODULES.keys()) == expected


class TestLeverage:
    def test_max_leverage_7_conditions(self):
        tier = LEVERAGE_TIERS[0]
        assert tier.conditions_met == 7
        assert tier.max_leverage == 100.0

    def test_no_leverage_below_3(self):
        tier = LEVERAGE_TIERS[-1]
        assert tier.conditions_met == 0
        assert tier.max_leverage == 2.0


class TestGauntlet:
    def test_7_levels(self):
        assert len(GAUNTLET_LEVELS) == 7

    def test_level_1_initiation(self):
        assert GAUNTLET_LEVELS[0].name == "INITIATION"

    def test_level_7_nemesis(self):
        assert GAUNTLET_LEVELS[6].name == "NEMESIS MODE"


class TestKelly:
    def test_quarter_kelly(self):
        assert KELLY_FRACTION == 0.25


class TestSessions:
    def test_5_session_windows(self):
        assert len(SESSION_WINDOWS) == 5

    def test_london_opens_at_8(self):
        london = [w for w in SESSION_WINDOWS if w.name == Session.LONDON][0]
        assert london.open_hour == 8
