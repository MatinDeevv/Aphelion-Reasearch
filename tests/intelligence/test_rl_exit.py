"""Tests for APHELION RL Exit Agent."""

import numpy as np
import pytest

from aphelion.intelligence.rl_exit.environment import (
    ExitEnvConfig,
    ExitEnvironment,
)
from aphelion.intelligence.rl_exit.agent import (
    ExitAgentConfig,
    RLExitAgent,
    ExitDecision,
    HAS_TORCH,
)


# ─── ExitEnvConfig tests ─────────────────────────────────────────────────────


class TestExitEnvConfig:

    def test_defaults(self):
        cfg = ExitEnvConfig()
        assert cfg.max_bars == 200
        assert cfg.hold_penalty < 0
        assert cfg.state_dim == 8 + cfg.feature_summary_dim

    def test_custom(self):
        cfg = ExitEnvConfig(max_bars=50, feature_summary_dim=4)
        assert cfg.max_bars == 50
        assert cfg.state_dim == 12


# ─── ExitEnvironment tests ───────────────────────────────────────────────────


class TestExitEnvironment:

    def test_reset_returns_state(self):
        env = ExitEnvironment()
        state = env.reset(
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            direction=1,
        )
        assert isinstance(state, np.ndarray)
        assert len(state) == env._cfg.state_dim

    def test_step_hold(self):
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        state, reward, done, info = env.step(0, 2351.0)
        assert done is False
        assert reward == env._cfg.hold_penalty
        assert info["bars_held"] == 1

    def test_step_agent_exit(self):
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        state, reward, done, info = env.step(1, 2355.0)
        assert done is True
        assert info["exit_reason"] == "AGENT_EXIT"
        assert reward > env._cfg.hold_penalty  # Should get R-multiple reward + bonus

    def test_sl_hit_terminates(self):
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        state, reward, done, info = env.step(0, 2339.0)  # Below SL
        assert done is True
        assert info["exit_reason"] == "SL_HIT"
        assert reward < 0

    def test_tp_hit_terminates(self):
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        state, reward, done, info = env.step(0, 2371.0)  # Above TP
        assert done is True
        assert info["exit_reason"] == "TP_HIT"
        assert reward > 0

    def test_max_bars_terminates(self):
        cfg = ExitEnvConfig(max_bars=3)
        env = ExitEnvironment(cfg)
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        env.step(0, 2351.0)
        env.step(0, 2352.0)
        state, reward, done, info = env.step(0, 2353.0)
        assert done is True
        assert info["exit_reason"] == "MAX_BARS"

    def test_short_direction(self):
        env = ExitEnvironment()
        state = env.reset(
            entry_price=2350.0,
            stop_loss=2360.0,  # SL above for SHORT
            take_profit=2330.0,  # TP below for SHORT
            direction=-1,
        )
        # Price drops = profit for short
        s, r, done, info = env.step(1, 2340.0)
        assert done is True
        assert info["r_multiple"] > 0

    def test_step_after_done_raises(self):
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        env.step(1, 2355.0)  # EXIT → done
        with pytest.raises(RuntimeError):
            env.step(0, 2356.0)

    def test_observe_builds_state(self):
        env = ExitEnvironment()
        state = env.observe(
            bars_held=10,
            current_price=2355.0,
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            direction=1,
        )
        assert isinstance(state, np.ndarray)
        assert len(state) == env._cfg.state_dim

    def test_r_multiple_calculation(self):
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        env._current_price = 2360.0
        r = env._current_r_multiple()
        assert abs(r - 1.0) < 0.01  # 10 profit / 10 risk = 1R

    def test_state_values_bounded(self):
        """State values should be clipped and normalised."""
        env = ExitEnvironment()
        env.reset(entry_price=2350.0, stop_loss=2340.0, take_profit=2370.0, direction=1)
        state, _, _, _ = env.step(0, 2355.0)
        # bars_held_normalised should be in [0, 1]
        assert 0 <= state[0] <= 1.0
        # direction should be +1 or -1
        assert state[6] in (1.0, -1.0)


# ─── ExitAgentConfig tests ───────────────────────────────────────────────────


class TestExitAgentConfig:

    def test_defaults(self):
        cfg = ExitAgentConfig()
        assert cfg.hidden_dim == 64
        assert cfg.gamma == 0.99
        assert cfg.clip_epsilon == 0.2

    def test_custom(self):
        cfg = ExitAgentConfig(hidden_dim=32, gamma=0.95)
        assert cfg.hidden_dim == 32
        assert cfg.gamma == 0.95


# ─── RLExitAgent tests ───────────────────────────────────────────────────────


class TestRLExitAgent:

    def test_init_no_crash(self):
        agent = RLExitAgent(state_dim=16)
        assert isinstance(agent, RLExitAgent)

    def test_act_returns_decision(self):
        agent = RLExitAgent(state_dim=16)
        state = np.random.default_rng(42).normal(0, 1, size=16).astype(np.float32)
        decision = agent.act(state)
        assert isinstance(decision, ExitDecision)
        assert decision.action in (0, 1)
        assert 0.0 <= decision.exit_probability <= 1.0
        assert 0.0 <= decision.confidence <= 1.0

    def test_rule_based_fallback(self):
        """Even without torch, agent should work via rule-based fallback."""
        cfg = ExitAgentConfig(fallback_r_exit=2.0)
        agent = RLExitAgent(state_dim=16, config=cfg)
        # State: [bars_frac=0.1, unrealised_R=2.5, ...]
        state = np.zeros(16, dtype=np.float32)
        state[0] = 0.1  # low bars
        state[1] = 2.5  # above 2R threshold
        decision = agent._rule_based_act(state)
        assert decision.action == 1  # Should exit at 2.5R

    def test_rule_based_hold(self):
        agent = RLExitAgent(state_dim=16)
        state = np.zeros(16, dtype=np.float32)
        state[0] = 0.1  # low bars
        state[1] = 0.5  # below threshold
        decision = agent._rule_based_act(state)
        assert decision.action == 0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_store_and_update(self):
        agent = RLExitAgent(state_dim=16, config=ExitAgentConfig(rollout_size=32, batch_size=16))
        env = ExitEnvironment(ExitEnvConfig(max_bars=20, feature_summary_dim=8))
        rng = np.random.default_rng(42)

        state = env.reset(2350.0, 2340.0, 2370.0, 1)
        for _ in range(50):
            decision = agent.act(state)
            price = 2350.0 + float(rng.normal(0, 2))
            next_state, reward, done, info = env.step(decision.action, price)
            agent.store_transition(state, decision.action, reward, next_state, done, decision)
            if done:
                state = env.reset(2350.0, 2340.0, 2370.0, 1)
            else:
                state = next_state

        stats = agent.update()
        assert "policy_loss" in stats
        assert stats.get("total_updates", 0) >= 1

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_deterministic_mode(self):
        cfg = ExitAgentConfig(deterministic=True, exit_threshold=0.5)
        agent = RLExitAgent(state_dim=16, config=cfg)
        state = np.zeros(16, dtype=np.float32)
        d1 = agent.act(state)
        d2 = agent.act(state)
        # In deterministic mode, same state → same action
        assert d1.action == d2.action

    def test_exit_decision_fields(self):
        d = ExitDecision(action=1, exit_probability=0.8, value_estimate=1.5, confidence=0.6, reason="test")
        assert d.action == 1
        assert d.exit_probability == 0.8
        assert d.reason == "test"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_update_insufficient_data(self):
        agent = RLExitAgent(state_dim=16, config=ExitAgentConfig(batch_size=100))
        stats = agent.update()
        assert stats.get("error") == "insufficient_data"
