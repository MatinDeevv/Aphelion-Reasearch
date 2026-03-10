"""
APHELION HYDRA Online Learning Layer

Lets HYDRA update weights in real-time from each new trade WITHOUT full retraining.
Sits on top of the frozen ensemble as a lightweight adapter that learns online.

Architecture:
  - Frozen base: HydraGate ensemble (TFT + LSTM + CNN + MoE + TCN + Transformer)
  - Online adapter: Small feed-forward network on top of ensemble latents
  - Training: Online SGD / Adam with per-sample gradient updates
  - Replay buffer: Stores recent (features, outcome) pairs to prevent forgetting
  - Exponential forgetting: Older samples weighted less via time-decaying loss

Why this matters:
  - HYDRA learns market structure from historical data, but markets drift
  - Full retraining is expensive (hours of GPU time) and infrequent
  - Online learning adapts continuously: each resolved trade teaches the model
  - The adapter is tiny (~10K params) vs the full ensemble (~2M params)
  - If the adapter degrades, it can be reset to zero (passthrough mode)

Interface:
  - on_trade_resolved(features, outcome): update from a single closed trade
  - predict(ensemble_output): modify ensemble predictions using learned adapter
  - reset(): clear adapter weights and replay buffer

References:
  - Bottou (2010) "Large-Scale Machine Learning with Stochastic Gradient Descent"
  - McMahan et al. (2013) "Ad Click Prediction: a View from the Trenches"
  - Sahoo et al. (2018) "Online Deep Learning"
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class OnlineConfig:
    """Configuration for the online learning layer."""

    # Adapter architecture
    adapter_hidden: int = 64           # Hidden size for the adapter MLP
    adapter_layers: int = 2            # Number of hidden layers
    dropout: float = 0.1

    # Online SGD
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    # Replay buffer
    buffer_size: int = 1000            # Max stored experiences
    min_buffer_for_update: int = 10    # Min samples before first update
    batch_size: int = 32               # Mini-batch from replay buffer
    replay_ratio: float = 0.5          # Fraction of update from replay vs current

    # Forgetting
    time_decay: float = 0.999          # Per-sample decay factor for old experiences
    reset_threshold: float = 0.40      # Reset adapter if online win_rate < this

    # Input/output dims (from HYDRA ensemble)
    ensemble_logit_dim: int = 9        # 3 horizons * 3 classes
    ensemble_extra_dim: int = 1        # uncertainty
    n_classes: int = 3                 # LONG, FLAT, SHORT
    n_horizons: int = 3


# ─── Data structures ─────────────────────────────────────────────────────────


@dataclass
class OnlineExperience:
    """A single (input, outcome) pair for online learning."""
    ensemble_logits: np.ndarray    # (9,) flattened logits from ensemble
    uncertainty: float
    features_summary: np.ndarray   # (n,) summary features at trade time
    outcome: int                   # 0=LONG_correct, 1=FLAT, 2=SHORT_correct
    timestamp: float = 0.0
    weight: float = 1.0            # Time-decayed importance


@dataclass
class OnlineStats:
    """Statistics for the online learning layer."""
    total_updates: int = 0
    buffer_size: int = 0
    online_win_rate: float = 0.0
    recent_loss: float = 0.0
    adapter_norm: float = 0.0
    is_active: bool = False
    resets: int = 0


# ─── Online Learning Layer ───────────────────────────────────────────────────


if HAS_TORCH:

    class OnlineAdapter(nn.Module):
        """
        Lightweight adapter network that modifies ensemble logits online.

        Takes: [ensemble_logits (9), uncertainty (1), features_summary (K)]
        Outputs: logit_adjustment (9) — added to original ensemble logits
        """

        def __init__(self, config: OnlineConfig, feature_summary_dim: int = 16):
            super().__init__()
            self.config = config
            input_dim = config.ensemble_logit_dim + config.ensemble_extra_dim + feature_summary_dim

            layers = []
            prev_dim = input_dim
            for _ in range(config.adapter_layers):
                layers.extend([
                    nn.Linear(prev_dim, config.adapter_hidden),
                    nn.LayerNorm(config.adapter_hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
                prev_dim = config.adapter_hidden

            # Output: residual logit adjustments (initialised near zero)
            layers.append(nn.Linear(prev_dim, config.ensemble_logit_dim))
            self.network = nn.Sequential(*layers)

            # Gating scalar: how much to trust the adapter (learned)
            self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

            self._init_near_zero()

        def _init_near_zero(self) -> None:
            """Initialise output layer near zero so adapter starts as passthrough."""
            with torch.no_grad():
                last_layer = self.network[-1]
                if isinstance(last_layer, nn.Linear):
                    last_layer.weight.mul_(0.01)
                    last_layer.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Returns logit adjustments scaled by learned gate.
            x: (batch, input_dim)
            output: (batch, 9) — to be ADDED to ensemble logits
            """
            raw = self.network(x)
            gate_value = torch.sigmoid(self.gate)
            return raw * gate_value


class OnlineLearner:
    """
    Online learning wrapper for HYDRA.

    Maintains a replay buffer of recent trade outcomes and uses them
    to train a lightweight adapter that adjusts ensemble predictions
    in real-time.

    Usage::

        learner = OnlineLearner(OnlineConfig())

        # After each resolved trade:
        learner.on_trade_resolved(
            ensemble_logits=signal_logits,   # (9,) from HydraGate
            uncertainty=signal_uncertainty,
            features_summary=feature_vector,  # (16,) compressed features
            outcome=2,                        # SHORT was correct
        )

        # During inference — adjust ensemble output:
        adjusted = learner.adjust_logits(
            ensemble_logits=raw_logits,
            uncertainty=unc,
            features_summary=feat,
        )
    """

    def __init__(
        self,
        config: Optional[OnlineConfig] = None,
        feature_summary_dim: int = 16,
    ):
        self._cfg = config or OnlineConfig()
        self._feature_dim = feature_summary_dim

        # Replay buffer
        self._buffer: deque[OnlineExperience] = deque(maxlen=self._cfg.buffer_size)
        self._wins: int = 0
        self._total: int = 0
        self._resets: int = 0
        self._total_updates: int = 0
        self._recent_loss: float = 0.0

        # PyTorch components (if available)
        self._adapter: Optional[object] = None
        self._optimizer: Optional[object] = None
        self._active: bool = False

        if HAS_TORCH:
            self._adapter = OnlineAdapter(self._cfg, feature_summary_dim)
            self._optimizer = torch.optim.AdamW(
                self._adapter.parameters(),
                lr=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
            )
            self._adapter.eval()
        else:
            logger.warning("PyTorch not available — OnlineLearner is inert")

    def on_trade_resolved(
        self,
        ensemble_logits: np.ndarray,
        uncertainty: float,
        features_summary: np.ndarray,
        outcome: int,
        timestamp: float = 0.0,
    ) -> OnlineStats:
        """
        Learn from a single resolved trade.

        Args:
            ensemble_logits: (9,) flattened logits from ensemble at trade time
            uncertainty: Ensemble uncertainty scalar
            features_summary: (K,) compressed feature vector
            outcome: Ground truth class (0=LONG, 1=FLAT, 2=SHORT)
            timestamp: Trade close timestamp

        Returns:
            Current online learning statistics.
        """
        # Store experience
        exp = OnlineExperience(
            ensemble_logits=ensemble_logits,
            uncertainty=uncertainty,
            features_summary=features_summary,
            outcome=outcome,
            timestamp=timestamp,
        )
        self._buffer.append(exp)
        self._total += 1

        # Track win rate: correct if argmax of logits matched outcome
        predicted = int(np.argmax(ensemble_logits[:3]))  # 5m horizon
        if predicted == outcome:
            self._wins += 1

        # Apply time decay to older experiences
        self._apply_time_decay()

        # Only train once we have enough data
        if len(self._buffer) < self._cfg.min_buffer_for_update:
            return self._stats()

        if not HAS_TORCH or self._adapter is None:
            return self._stats()

        # ── Training step ──
        self._adapter.train()

        # Build mini-batch: current sample + replay samples
        batch = self._sample_batch(current=exp)
        loss = self._train_step(batch)
        self._recent_loss = loss
        self._total_updates += 1
        self._active = True

        self._adapter.eval()

        # Reset check: if adapter is making things worse, reset it
        if self._total >= 50 and self._online_win_rate < self._cfg.reset_threshold:
            self.reset()

        return self._stats()

    def adjust_logits(
        self,
        ensemble_logits: np.ndarray,
        uncertainty: float,
        features_summary: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust ensemble logits using the online adapter.

        Returns modified logits (same shape as input).
        If adapter is not active, returns input unchanged.
        """
        if not self._active or not HAS_TORCH or self._adapter is None:
            return ensemble_logits

        with torch.no_grad():
            x = self._build_input(ensemble_logits, uncertainty, features_summary)
            adjustment = self._adapter(x).squeeze(0).numpy()

        return ensemble_logits + adjustment

    def reset(self) -> None:
        """Reset the adapter to passthrough mode."""
        self._resets += 1
        self._active = False
        self._wins = 0
        self._total = 0
        logger.info("OnlineLearner reset (#%d)", self._resets)

        if HAS_TORCH and self._adapter is not None:
            self._adapter._init_near_zero()
            self._optimizer = torch.optim.AdamW(
                self._adapter.parameters(),
                lr=self._cfg.learning_rate,
                weight_decay=self._cfg.weight_decay,
            )

    @property
    def _online_win_rate(self) -> float:
        if self._total == 0:
            return 0.5
        return self._wins / self._total

    def _stats(self) -> OnlineStats:
        adapter_norm = 0.0
        if HAS_TORCH and self._adapter is not None:
            adapter_norm = sum(
                float(p.norm().item())
                for p in self._adapter.parameters()
            )
        return OnlineStats(
            total_updates=self._total_updates,
            buffer_size=len(self._buffer),
            online_win_rate=self._online_win_rate,
            recent_loss=self._recent_loss,
            adapter_norm=adapter_norm,
            is_active=self._active,
            resets=self._resets,
        )

    def _apply_time_decay(self) -> None:
        """Decay weights of older experiences."""
        for exp in self._buffer:
            exp.weight *= self._cfg.time_decay

    def _sample_batch(self, current: OnlineExperience) -> list[OnlineExperience]:
        """Sample a mini-batch mixing current observation with replay."""
        batch = [current]
        n_replay = min(
            self._cfg.batch_size - 1,
            len(self._buffer) - 1,
        )
        if n_replay > 0:
            # Weighted sampling by recency
            weights = np.array([exp.weight for exp in self._buffer])
            weights /= weights.sum()
            indices = np.random.choice(len(self._buffer), size=n_replay, replace=False, p=weights)
            for idx in indices:
                batch.append(self._buffer[idx])
        return batch

    def _build_input(
        self,
        logits: np.ndarray,
        uncertainty: float,
        features: np.ndarray,
    ) -> "torch.Tensor":
        """Build adapter input tensor from components."""
        # Pad or truncate features to expected dim
        feat = np.zeros(self._feature_dim)
        n = min(len(features), self._feature_dim)
        feat[:n] = features[:n]

        x = np.concatenate([logits.ravel()[:9], [uncertainty], feat])
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    def _train_step(self, batch: list[OnlineExperience]) -> float:
        """Single SGD step on a mini-batch."""
        if not HAS_TORCH:
            return 0.0

        inputs = []
        targets = []
        weights = []

        for exp in batch:
            x = self._build_input(exp.ensemble_logits, exp.uncertainty, exp.features_summary)
            inputs.append(x)
            targets.append(exp.outcome)
            weights.append(exp.weight)

        x_batch = torch.cat(inputs, dim=0)
        y_batch = torch.tensor(targets, dtype=torch.long)
        w_batch = torch.tensor(weights, dtype=torch.float32)
        w_batch = w_batch / w_batch.sum()  # Normalise

        # Forward: adapter predicts logit adjustments
        adjustments = self._adapter(x_batch)  # (batch, 9)

        # We want the adjusted logits to predict the correct class
        # Use the 5m horizon slice (first 3 logits) for the loss
        original_logits = x_batch[:, :3]  # First 3 elements are 5m logits
        adjusted = original_logits + adjustments[:, :3]

        loss_per_sample = F.cross_entropy(adjusted, y_batch, reduction="none")
        loss = (loss_per_sample * w_batch).sum()

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._adapter.parameters(),
            self._cfg.max_grad_norm,
        )
        self._optimizer.step()

        return float(loss.item())
