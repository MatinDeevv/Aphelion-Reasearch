"""
APHELION RL Exit — PPO Agent for optimal exit timing.

A Proximal Policy Optimisation agent that learns exit policies from trade
experience. Supports both training (with rollout buffer and gradient updates)
and inference (greedy or stochastic action selection).

Architecture:
  - Shared feature encoder (2xFC + LayerNorm + GELU)
  - Policy head → Categorical(HOLD, EXIT) with log-probabilities
  - Value head → scalar baseline for variance reduction

The agent is intentionally small (~5K params) to:
  1. Avoid overfitting on small trade datasets
  2. Enable fast inference on every bar of an open trade
  3. Be easily serialisable for paper trading deployment

Numpy-only fallback: if PyTorch is unavailable, a simple rule-based
agent mimics basic exit behaviour (exit at 2R or after N bars).

References:
  - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
  - Barto, Bradtke & Singh (1995) "Learning to Act using Real-Time DP"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ExitAgentConfig:
    """PPO agent configuration."""

    # Network
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    dropout: float = 0.05

    # PPO hyperparameters
    gamma: float = 0.99                 # Discount factor
    gae_lambda: float = 0.95           # GAE lambda
    clip_epsilon: float = 0.2          # PPO clip range
    entropy_coeff: float = 0.01        # Entropy bonus for exploration
    value_coeff: float = 0.5           # Value loss weight
    max_grad_norm: float = 0.5

    # Training
    learning_rate: float = 3e-4
    n_epochs: int = 4                  # PPO epochs per batch
    batch_size: int = 64
    rollout_size: int = 256            # Transitions per rollout before update

    # Inference
    deterministic: bool = False        # Greedy mode for deployment
    exit_threshold: float = 0.55       # Exit only if P(exit) > threshold

    # Rule-based fallback (when no torch)
    fallback_r_exit: float = 2.0       # Exit at 2R
    fallback_max_bars: int = 100


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class ExitDecision:
    """Output of the RL exit agent."""
    action: int                    # 0=HOLD, 1=EXIT
    exit_probability: float        # P(EXIT)
    value_estimate: float          # V(s) from value head
    confidence: float              # Calibrated confidence in the decision
    reason: str = ""               # Human-readable explanation


@dataclass
class Transition:
    """A single environment transition for the rollout buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


# ─── PPO Network (torch) ─────────────────────────────────────────────────────


if HAS_TORCH:

    class ExitPolicyNetwork(nn.Module):
        """
        Actor-Critic network for the exit agent.

        Shared trunk → policy head (2 actions) + value head (scalar).
        """

        def __init__(self, state_dim: int, config: ExitAgentConfig):
            super().__init__()
            self.config = config

            # Shared feature encoder
            layers = []
            prev_dim = state_dim
            for _ in range(config.n_hidden_layers):
                layers.extend([
                    nn.Linear(prev_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
                prev_dim = config.hidden_dim
            self.trunk = nn.Sequential(*layers)

            # Policy head: HOLD / EXIT
            self.policy_head = nn.Linear(config.hidden_dim, 2)

            # Value head: V(s)
            self.value_head = nn.Linear(config.hidden_dim, 1)

        def forward(self, state: torch.Tensor):
            """
            Returns:
                logits: (batch, 2)
                value: (batch, 1)
            """
            features = self.trunk(state)
            logits = self.policy_head(features)
            value = self.value_head(features)
            return logits, value


# ─── PPO Agent ────────────────────────────────────────────────────────────────


class RLExitAgent:
    """
    PPO-based exit agent.

    Training usage::

        agent = RLExitAgent(state_dim=16, config=ExitAgentConfig())
        env = ExitEnvironment()

        # Collect rollouts
        state = env.reset(...)
        for _ in range(config.rollout_size):
            decision = agent.act(state)
            next_state, reward, done, info = env.step(decision.action, price)
            agent.store_transition(state, decision.action, reward,
                                   next_state, done, decision)
            state = next_state if not done else env.reset(...)

        # Update policy
        stats = agent.update()

    Inference usage::

        decision = agent.act(state)
        if decision.action == 1:
            close_position()
    """

    def __init__(
        self,
        state_dim: int = 16,
        config: Optional[ExitAgentConfig] = None,
    ):
        self._cfg = config or ExitAgentConfig()
        self._state_dim = state_dim

        # Rollout buffer
        self._buffer: list[Transition] = []
        self._total_updates: int = 0
        self._total_episodes: int = 0

        # Network
        self._network: Optional[object] = None
        self._optimizer: Optional[object] = None

        if HAS_TORCH:
            self._network = ExitPolicyNetwork(state_dim, self._cfg)
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._cfg.learning_rate,
            )
            self._network.eval()
        else:
            logger.warning("PyTorch not available — using rule-based exit fallback")

    def act(self, state: np.ndarray) -> ExitDecision:
        """
        Choose an action given the current state.

        Works in both torch and numpy-only modes.
        """
        if not HAS_TORCH or self._network is None:
            return self._rule_based_act(state)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self._network(s)
            probs = F.softmax(logits, dim=-1).squeeze(0)

            exit_prob = float(probs[1].item())
            val = float(value.item())

            if self._cfg.deterministic:
                action = 1 if exit_prob > self._cfg.exit_threshold else 0
            else:
                dist = Categorical(probs)
                action = int(dist.sample().item())

        # Confidence: distance from 0.5 (max uncertainty)
        confidence = abs(exit_prob - 0.5) * 2.0

        reason = ""
        if action == 1:
            reason = f"EXIT (p={exit_prob:.2f}, V={val:.3f})"
        else:
            reason = f"HOLD (p_exit={exit_prob:.2f}, V={val:.3f})"

        return ExitDecision(
            action=action,
            exit_probability=exit_prob,
            value_estimate=val,
            confidence=confidence,
            reason=reason,
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        decision: ExitDecision,
    ) -> None:
        """Store a transition in the rollout buffer."""
        if not HAS_TORCH:
            return

        # Compute log_prob for the taken action
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = self._network(s)
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = float(log_probs[0, action].item())

        self._buffer.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=decision.value_estimate,
        ))

        if done:
            self._total_episodes += 1

    def update(self) -> dict:
        """
        Run PPO update on collected rollout buffer.

        Returns:
            Training statistics dict.
        """
        if not HAS_TORCH or self._network is None:
            return {"error": "no_torch"}

        if len(self._buffer) < self._cfg.batch_size:
            return {"error": "insufficient_data", "buffer_size": len(self._buffer)}

        # ── Compute returns and advantages (GAE) ──
        states, actions, old_log_probs, returns, advantages = self._compute_gae()

        self._network.train()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0

        for _ in range(self._cfg.n_epochs):
            # Mini-batch iteration
            indices = np.random.permutation(len(states))
            for start in range(0, len(indices), self._cfg.batch_size):
                end = start + self._cfg.batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Normalise advantages
                if len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                logits, values = self._network(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self._cfg.clip_epsilon, 1 + self._cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                # Combined loss
                loss = (
                    policy_loss
                    + self._cfg.value_coeff * value_loss
                    - self._cfg.entropy_coeff * entropy
                )

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    self._cfg.max_grad_norm,
                )
                self._optimizer.step()

                # Accumulate stats
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float(approx_kl.item())
                n_updates += 1

        self._network.eval()
        self._total_updates += 1

        # Clear buffer
        self._buffer.clear()

        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates

        stats["total_updates"] = self._total_updates
        stats["total_episodes"] = self._total_episodes

        return stats

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        if HAS_TORCH and self._network is not None:
            torch.save({
                "network": self._network.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "total_updates": self._total_updates,
                "total_episodes": self._total_episodes,
                "config": self._cfg,
            }, path)
            logger.info("RLExitAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        if not HAS_TORCH:
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if self._network is not None:
            self._network.load_state_dict(checkpoint["network"])
            self._network.eval()
        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._total_updates = checkpoint.get("total_updates", 0)
        self._total_episodes = checkpoint.get("total_episodes", 0)
        logger.info("RLExitAgent loaded from %s", path)

    def _compute_gae(self):
        """Compute Generalised Advantage Estimation from buffer."""
        n = len(self._buffer)
        states_np = np.array([t.state for t in self._buffer], dtype=np.float32)
        actions_np = np.array([t.action for t in self._buffer], dtype=np.int64)
        old_lp_np = np.array([t.log_prob for t in self._buffer], dtype=np.float32)

        # Bootstrap last value
        rewards = [t.reward for t in self._buffer]
        values = [t.value for t in self._buffer]
        dones = [t.done for t in self._buffer]

        # Append bootstrap value for last state
        with torch.no_grad():
            last_s = torch.tensor(self._buffer[-1].next_state, dtype=torch.float32).unsqueeze(0)
            _, last_v = self._network(last_s)
            next_value = float(last_v.item()) if not dones[-1] else 0.0

        # GAE computation
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self._cfg.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self._cfg.gamma * self._cfg.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns_np = advantages + np.array(values, dtype=np.float32)

        # Convert to tensors
        states_t = torch.tensor(states_np)
        actions_t = torch.tensor(actions_np)
        old_lp_t = torch.tensor(old_lp_np)
        returns_t = torch.tensor(returns_np)
        advantages_t = torch.tensor(advantages)

        return states_t, actions_t, old_lp_t, returns_t, advantages_t

    def _rule_based_act(self, state: np.ndarray) -> ExitDecision:
        """
        Simple rule-based exit for environments without PyTorch.

        Uses the state vector convention from ExitEnvironment:
          state[0] = bars_held_normalised
          state[1] = unrealised_R
        """
        bars_frac = state[0] if len(state) > 0 else 0.0
        r_mult = state[1] if len(state) > 1 else 0.0

        # Exit at target R or after too many bars
        exit = False
        reason = "HOLD (rule-based)"

        if r_mult >= self._cfg.fallback_r_exit:
            exit = True
            reason = f"EXIT: R={r_mult:.1f} >= {self._cfg.fallback_r_exit}"
        elif bars_frac > 0.8:
            exit = True
            reason = f"EXIT: bars_frac={bars_frac:.2f} > 0.8"
        elif r_mult < -0.5:
            # Losing trade, let SL handle it
            pass

        return ExitDecision(
            action=1 if exit else 0,
            exit_probability=0.9 if exit else 0.1,
            value_estimate=r_mult,
            confidence=0.5,  # Rule-based → moderate confidence
            reason=reason,
        )
