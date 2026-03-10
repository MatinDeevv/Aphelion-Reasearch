"""
APHELION RL Exit — Gym-like Environment for exit timing.

Models a single open trade as an episodic MDP:
  - State: (bars_held, unrealised_R, volatility_ratio, momentum, distance_to_SL,
            distance_to_TP, regime_code, trend_strength, feature_summary)
  - Actions: 0 = HOLD, 1 = EXIT_NOW
  - Reward: R-multiple at exit (shaped with small HOLD penalty to discourage idling)
  - Termination: agent chooses EXIT, SL hit, TP hit, or max_bars reached

Why R-multiple reward?
  - Makes learning transferable across instruments / price levels
  - Directly tied to risk management (1R = initial risk unit)
  - Sharpe-contribution shaping would couple episodes; R-multiples are cleaner

References:
  - Schulman et al. (2017) "Proximal Policy Optimization"
  - Deng et al. (2016) "Deep Direct Reinforcement Learning for Financial Signal
    Representation and Trading"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExitEnvConfig:
    """Configuration for the exit-timing environment."""

    # State dimensions
    max_bars: int = 200               # Max bars before forced exit
    feature_summary_dim: int = 8      # Compressed market feature vector

    # Reward shaping
    hold_penalty: float = -0.001      # Small cost per bar to discourage idling
    early_exit_bonus: float = 0.1     # Bonus for exiting profitably before TP
    sl_penalty: float = -0.2          # Additional penalty when SL hit (on top of -1R)
    tp_bonus: float = 0.1             # Bonus when TP hit
    r_multiple_scale: float = 1.0     # Scale factor on R-multiple reward

    # Normalisation
    max_r_clip: float = 5.0           # Clip R-multiples to avoid extreme rewards
    vol_window: int = 20              # Lookback for volatility ratio calc

    @property
    def state_dim(self) -> int:
        """Total state vector dimension."""
        return 8 + self.feature_summary_dim  # 8 core + features


class ExitEnvironment:
    """
    Gym-like environment for training the RL exit agent.

    One episode = one open trade, from entry to exit.
    The agent observes market state at each bar and decides HOLD or EXIT.

    Usage (training loop)::

        env = ExitEnvironment(ExitEnvConfig())

        state = env.reset(
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            direction=1,  # LONG
        )

        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action, bar_data)

    Usage (live inference)::

        state = env.observe(
            bars_held=12,
            current_price=2356.0,
            entry_price=2350.0,
            stop_loss=2340.0,
            take_profit=2370.0,
            direction=1,
            features_summary=feat_vec,
        )
        action = agent.act(state)
    """

    def __init__(self, config: Optional[ExitEnvConfig] = None):
        self._cfg = config or ExitEnvConfig()

        # Episode state
        self._entry_price: float = 0.0
        self._stop_loss: float = 0.0
        self._take_profit: float = 0.0
        self._direction: int = 1       # +1 LONG, -1 SHORT
        self._bars_held: int = 0
        self._current_price: float = 0.0
        self._done: bool = True
        self._initial_risk: float = 1.0

        # Rolling volatility tracker
        self._price_history: list[float] = []

    def reset(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: int,
        features_summary: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Start a new episode for an open trade.

        Returns:
            Initial state vector.
        """
        self._entry_price = entry_price
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._direction = direction
        self._current_price = entry_price
        self._bars_held = 0
        self._done = False
        self._initial_risk = abs(entry_price - stop_loss)
        if self._initial_risk < 1e-8:
            self._initial_risk = entry_price * 0.005  # Fallback: 0.5%
        self._price_history = [entry_price]

        feat = features_summary if features_summary is not None else np.zeros(self._cfg.feature_summary_dim)
        return self._build_state(feat)

    def step(
        self,
        action: int,
        current_price: float,
        features_summary: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance one bar and apply the agent's action.

        Args:
            action: 0=HOLD, 1=EXIT
            current_price: Current market price
            features_summary: (K,) compressed feature vector

        Returns:
            (next_state, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode already done — call reset()")

        self._bars_held += 1
        self._current_price = current_price
        self._price_history.append(current_price)

        feat = features_summary if features_summary is not None else np.zeros(self._cfg.feature_summary_dim)

        r_mult = self._current_r_multiple()
        info: dict = {"bars_held": self._bars_held, "r_multiple": r_mult, "exit_reason": ""}

        # ── Check natural exits first ──
        # Stop loss hit
        if self._is_sl_hit(current_price):
            reward = self._cfg.r_multiple_scale * np.clip(r_mult, -self._cfg.max_r_clip, self._cfg.max_r_clip)
            reward += self._cfg.sl_penalty
            self._done = True
            info["exit_reason"] = "SL_HIT"
            return self._build_state(feat), reward, True, info

        # Take profit hit
        if self._is_tp_hit(current_price):
            reward = self._cfg.r_multiple_scale * np.clip(r_mult, -self._cfg.max_r_clip, self._cfg.max_r_clip)
            reward += self._cfg.tp_bonus
            self._done = True
            info["exit_reason"] = "TP_HIT"
            return self._build_state(feat), reward, True, info

        # Max bars reached — forced exit
        if self._bars_held >= self._cfg.max_bars:
            reward = self._cfg.r_multiple_scale * np.clip(r_mult, -self._cfg.max_r_clip, self._cfg.max_r_clip)
            self._done = True
            info["exit_reason"] = "MAX_BARS"
            return self._build_state(feat), reward, True, info

        # ── Agent-chosen exit ──
        if action == 1:
            reward = self._cfg.r_multiple_scale * np.clip(r_mult, -self._cfg.max_r_clip, self._cfg.max_r_clip)
            if r_mult > 0:
                reward += self._cfg.early_exit_bonus
            self._done = True
            info["exit_reason"] = "AGENT_EXIT"
            return self._build_state(feat), reward, True, info

        # HOLD — small penalty to discourage idling
        reward = self._cfg.hold_penalty
        return self._build_state(feat), reward, False, info

    def observe(
        self,
        bars_held: int,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: int,
        features_summary: Optional[np.ndarray] = None,
        price_history: Optional[list[float]] = None,
    ) -> np.ndarray:
        """
        Build a state vector for live inference (no episode management).

        Used at runtime when we just need a decision on an existing position.
        """
        self._entry_price = entry_price
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._direction = direction
        self._bars_held = bars_held
        self._current_price = current_price
        self._initial_risk = abs(entry_price - stop_loss)
        if self._initial_risk < 1e-8:
            self._initial_risk = entry_price * 0.005
        self._price_history = price_history if price_history else [entry_price, current_price]

        feat = features_summary if features_summary is not None else np.zeros(self._cfg.feature_summary_dim)
        return self._build_state(feat)

    def _build_state(self, features_summary: np.ndarray) -> np.ndarray:
        """
        Assemble the state vector.

        [0] bars_held_normalised   (0..1, relative to max_bars)
        [1] unrealised_R           (current R-multiple, clipped)
        [2] volatility_ratio       (recent vol / initial risk)
        [3] momentum               (price change over last 5 bars / initial_risk)
        [4] distance_to_SL         (signed, normalised by initial_risk)
        [5] distance_to_TP         (signed, normalised by initial_risk)
        [6] direction              (+1 or -1)
        [7] time_pressure          (non-linear urgency as bars → max)
        [8..] features_summary
        """
        r_mult = self._current_r_multiple()

        # Volatility ratio
        if len(self._price_history) >= 2:
            returns = np.diff(self._price_history[-self._cfg.vol_window:])
            vol = float(np.std(returns)) if len(returns) > 1 else 0.0
        else:
            vol = 0.0
        vol_ratio = vol / self._initial_risk if self._initial_risk > 0 else 0.0

        # Momentum (5-bar)
        if len(self._price_history) >= 6:
            momentum = (self._price_history[-1] - self._price_history[-6]) / self._initial_risk
        else:
            momentum = 0.0

        # Distances to SL/TP normalised by initial risk
        if self._direction == 1:  # LONG
            dist_sl = (self._current_price - self._stop_loss) / self._initial_risk
            dist_tp = (self._take_profit - self._current_price) / self._initial_risk
        else:  # SHORT
            dist_sl = (self._stop_loss - self._current_price) / self._initial_risk
            dist_tp = (self._current_price - self._take_profit) / self._initial_risk

        # Time pressure: increases non-linearly as we approach max_bars
        t_frac = self._bars_held / max(self._cfg.max_bars, 1)
        time_pressure = t_frac ** 2  # Quadratic urgency

        core = np.array([
            t_frac,
            np.clip(r_mult, -self._cfg.max_r_clip, self._cfg.max_r_clip),
            np.clip(vol_ratio, 0.0, 5.0),
            np.clip(momentum, -5.0, 5.0),
            np.clip(dist_sl, -5.0, 10.0),
            np.clip(dist_tp, -5.0, 10.0),
            float(self._direction),
            time_pressure,
        ], dtype=np.float32)

        # Pad or truncate features
        feat = np.zeros(self._cfg.feature_summary_dim, dtype=np.float32)
        n = min(len(features_summary), self._cfg.feature_summary_dim)
        feat[:n] = features_summary[:n]

        return np.concatenate([core, feat])

    def _current_r_multiple(self) -> float:
        """Unrealised R-multiple of the current position."""
        if self._initial_risk < 1e-8:
            return 0.0
        pnl = (self._current_price - self._entry_price) * self._direction
        return pnl / self._initial_risk

    def _is_sl_hit(self, price: float) -> bool:
        if self._direction == 1:
            return price <= self._stop_loss
        else:
            return price >= self._stop_loss

    def _is_tp_hit(self, price: float) -> bool:
        if self._direction == 1:
            return price >= self._take_profit
        else:
            return price <= self._take_profit
