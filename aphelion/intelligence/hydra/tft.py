"""
APHELION HYDRA — Temporal Fusion Transformer
Full TFT implementation for multi-horizon XAU/USD direction prediction.
Architecture: VSN → GRN → LSTM encoder → Interpretable Multi-Head Attention → Output heads

Outputs per bar:
  - P(LONG) / P(SHORT) / P(FLAT) for 5m, 15m, 1h horizons
  - Quantile forecasts (P10/P50/P90) for each horizon
  - Calibrated uncertainty estimate
  - Feature importance weights (interpretable)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class TFTConfig:
    """Temporal Fusion Transformer hyperparameters."""
    # Input dimensions
    n_continuous: int = len(CONTINUOUS_FEATURES)
    n_categorical: int = len(CATEGORICAL_FEATURES)
    cat_embedding_dims: list[int] = field(default_factory=lambda: [8, 8])  # per-cat embedding size
    cat_cardinalities: list[int] = field(default_factory=lambda: [5, 7])   # session(5), day(7)

    # Model dimensions
    hidden_dim: int = 256
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1

    # Sequence
    lookback: int = 64

    # Outputs
    n_horizons: int = 3               # 5m, 15m, 1h
    n_classes: int = 3                # SHORT, FLAT, LONG
    n_quantiles: int = 3              # P10, P50, P90
    quantile_targets: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


# ─── Building Blocks ────────────────────────────────────────────────────────


if HAS_TORCH:

    class GatedLinearUnit(nn.Module):
        """Gated Linear Unit: splits input, applies sigmoid gate."""

        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
            self.gate = nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x) * torch.sigmoid(self.gate(x))

    class GatedResidualNetwork(nn.Module):
        """
        Gated Residual Network (GRN).
        Non-linear processing with skip connections and gating.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            context_dim: Optional[int] = None,
        ):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

            if context_dim is not None:
                self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
            else:
                self.context_proj = None

            self.glu = GatedLinearUnit(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_dim)

            # Skip connection (project if dimensions differ)
            if input_dim != output_dim:
                self.skip_proj = nn.Linear(input_dim, output_dim)
            else:
                self.skip_proj = None

        def forward(
            self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            residual = x
            if self.skip_proj is not None:
                residual = self.skip_proj(residual)

            hidden = self.fc1(x)
            if self.context_proj is not None and context is not None:
                hidden = hidden + self.context_proj(context)
            hidden = self.elu(hidden)
            hidden = self.fc2(hidden)
            hidden = self.dropout(hidden)
            hidden = self.glu(hidden)

            return self.layer_norm(hidden + residual)

    class VariableSelectionNetwork(nn.Module):
        """
        Variable Selection Network (VSN).
        Learns per-timestep feature importance via softmax weighting.
        """

        def __init__(
            self,
            input_dim: int,
            n_vars: int,
            hidden_dim: int,
            dropout: float = 0.1,
            context_dim: Optional[int] = None,
        ):
            super().__init__()
            self.n_vars = n_vars

            # Per-variable GRNs
            self.var_grns = nn.ModuleList([
                GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
                for _ in range(n_vars)
            ])

            # Softmax feature selection weights
            grn_input = n_vars * hidden_dim
            self.selection_grn = GatedResidualNetwork(
                grn_input, hidden_dim, n_vars, dropout, context_dim,
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(
            self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                inputs: (batch, seq_len, n_vars, var_dim) or (batch, n_vars, var_dim)
                context: optional static context

            Returns:
                (selected_output, selection_weights)
            """
            # Handle both 3D and 4D input
            has_time = inputs.dim() == 4
            if has_time:
                batch, seq_len, n_vars, var_dim = inputs.shape
                inputs_flat = inputs.reshape(batch * seq_len, n_vars, var_dim)
            else:
                batch, n_vars, var_dim = inputs.shape
                inputs_flat = inputs

            # Process each variable through its GRN
            var_outputs = []
            for i in range(self.n_vars):
                var_outputs.append(self.var_grns[i](inputs_flat[:, i, :]))

            var_outputs = torch.stack(var_outputs, dim=1)  # (B, n_vars, hidden)

            # Concatenate for selection
            B = var_outputs.shape[0]
            flat = var_outputs.reshape(B, -1)  # (B, n_vars * hidden)

            # Selection weights
            ctx = None
            if context is not None and has_time:
                ctx = context.unsqueeze(1).expand(-1, seq_len, -1).reshape(B, -1) if context.dim() == 2 else context
            elif context is not None:
                ctx = context

            weights = self.selection_grn(flat, ctx)   # (B, n_vars)
            weights = self.softmax(weights)            # (B, n_vars)

            # Weighted combination
            selected = torch.sum(
                var_outputs * weights.unsqueeze(-1), dim=1,
            )  # (B, hidden)

            if has_time:
                selected = selected.reshape(batch, seq_len, -1)
                weights = weights.reshape(batch, seq_len, -1)

            return selected, weights

    class InterpretableMultiHeadAttention(nn.Module):
        """
        Interpretable Multi-Head Attention.
        Uses a single value head so attention weights are directly interpretable.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, self.d_k)  # Single value head
            self.W_o = nn.Linear(self.d_k, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch, seq_len, _ = query.shape

            # Multi-head Q, K
            Q = self.W_q(query).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value)  # (batch, seq_len, d_k) — single head

            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Average attention across heads for interpretability
            avg_attn = attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)

            # Apply attention to single value head
            context = torch.matmul(avg_attn, V)  # (batch, seq_len, d_k)
            output = self.W_o(context)

            return output, avg_attn

    class TemporalFusionTransformer(nn.Module):
        """
        Full Temporal Fusion Transformer for XAU/USD direction prediction.

        Architecture:
          1. Input embeddings (continuous projection + categorical embeddings)
          2. Variable Selection Networks (per-timestep feature importance)
          3. LSTM encoder (temporal patterns)
          4. Static enrichment via GRN
          5. Interpretable Multi-Head Attention (long-range dependencies)
          6. Position-wise feedforward
          7. Output heads:
             - Classification: P(SHORT)/P(FLAT)/P(LONG) per horizon
             - Quantile regression: P10/P50/P90 return forecasts
             - Uncertainty: calibrated aleatoric uncertainty
        """

        def __init__(self, config: Optional[TFTConfig] = None):
            super().__init__()
            self.config = config or TFTConfig()
            cfg = self.config

            # ── Input Embeddings ──────────────────────────────────────────
            # Project each continuous feature to hidden_dim individually
            self.cont_projections = nn.ModuleList([
                nn.Linear(1, cfg.hidden_dim) for _ in range(cfg.n_continuous)
            ])

            # Categorical embeddings
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, emb_dim)
                for card, emb_dim in zip(cfg.cat_cardinalities, cfg.cat_embedding_dims)
            ])

            # Project categorical embeddings to hidden_dim
            self.cat_projections = nn.ModuleList([
                nn.Linear(emb_dim, cfg.hidden_dim)
                for emb_dim in cfg.cat_embedding_dims
            ])

            total_vars = cfg.n_continuous + cfg.n_categorical

            # ── Variable Selection ────────────────────────────────────────
            self.vsn = VariableSelectionNetwork(
                input_dim=cfg.hidden_dim,
                n_vars=total_vars,
                hidden_dim=cfg.hidden_dim,
                dropout=cfg.dropout,
            )

            # ── LSTM Encoder ──────────────────────────────────────────────
            self.lstm = nn.LSTM(
                input_size=cfg.hidden_dim,
                hidden_size=cfg.hidden_dim,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.lstm_layers > 1 else 0,
                bidirectional=False,
            )

            # ── Post-LSTM GRN ─────────────────────────────────────────────
            self.post_lstm_grn = GatedResidualNetwork(
                cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout,
            )
            self.post_lstm_norm = nn.LayerNorm(cfg.hidden_dim)

            # ── Multi-Head Attention ──────────────────────────────────────
            self.attention = InterpretableMultiHeadAttention(
                cfg.hidden_dim, cfg.attention_heads, cfg.dropout,
            )
            self.post_attn_grn = GatedResidualNetwork(
                cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.dropout,
            )
            self.post_attn_norm = nn.LayerNorm(cfg.hidden_dim)

            # ── Position-wise Feedforward ─────────────────────────────────
            self.ff_grn = GatedResidualNetwork(
                cfg.hidden_dim, cfg.hidden_dim * 4, cfg.hidden_dim, cfg.dropout,
            )

            # ── Output Heads ──────────────────────────────────────────────
            # Classification: 3 classes per horizon
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_dim // 2, cfg.n_classes),
                )
                for _ in range(cfg.n_horizons)
            ])

            # Quantile regression: 3 quantiles per horizon
            self.quantile_heads = nn.ModuleList([
                nn.Linear(cfg.hidden_dim, cfg.n_quantiles)
                for _ in range(cfg.n_horizons)
            ])

            # Uncertainty head: single scalar
            self.uncertainty_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim // 4, 1),
                nn.Softplus(),  # Ensure positive uncertainty
            )

            self._init_weights()

        def _init_weights(self):
            """Xavier uniform initialization for stable training."""
            for name, param in self.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        def forward(
            self,
            cont_inputs: torch.Tensor,     # (batch, seq_len, n_continuous)
            cat_inputs: torch.Tensor,      # (batch, seq_len, n_categorical)
        ) -> dict[str, torch.Tensor]:
            """
            Forward pass through the full TFT.

            Returns dict with:
                - logits_5m, logits_15m, logits_1h: (batch, 3) raw class logits
                - probs_5m, probs_15m, probs_1h: (batch, 3) softmax probabilities
                - quantiles_5m, quantiles_15m, quantiles_1h: (batch, 3) P10/P50/P90
                - uncertainty: (batch, 1) aleatoric uncertainty
                - feature_weights: (batch, seq_len, n_vars) VSN importance weights
                - attention_weights: (batch, seq_len, seq_len) attention map
            """
            cfg = self.config
            batch, seq_len, _ = cont_inputs.shape

            # ── 1. Input Embeddings ───────────────────────────────────────
            # Project each continuous feature individually
            cont_embedded = []
            for i, proj in enumerate(self.cont_projections):
                cont_embedded.append(proj(cont_inputs[:, :, i:i+1]))
            # (batch, seq_len, n_cont, hidden)

            cat_embedded = []
            for i, (emb, proj) in enumerate(zip(self.cat_embeddings, self.cat_projections)):
                cat_emb = emb(cat_inputs[:, :, i])    # (batch, seq_len, emb_dim)
                cat_embedded.append(proj(cat_emb))     # (batch, seq_len, hidden)

            # Stack all variables: (batch, seq_len, total_vars, hidden)
            all_vars = torch.stack(cont_embedded + cat_embedded, dim=2)

            # ── 2. Variable Selection ─────────────────────────────────────
            selected, var_weights = self.vsn(all_vars)
            # selected: (batch, seq_len, hidden)
            # var_weights: (batch, seq_len, n_vars)

            # ── 3. LSTM Encoder ───────────────────────────────────────────
            lstm_out, _ = self.lstm(selected)
            # (batch, seq_len, hidden)

            # Post-LSTM processing with skip connection
            lstm_processed = self.post_lstm_grn(lstm_out)
            lstm_processed = self.post_lstm_norm(lstm_processed + selected)

            # ── 4. Multi-Head Attention ───────────────────────────────────
            # Causal mask to prevent attending to future
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=cont_inputs.device),
            ).unsqueeze(0)

            attn_out, attn_weights = self.attention(
                lstm_processed, lstm_processed, lstm_processed, causal_mask,
            )

            # Post-attention processing with skip connection
            attn_processed = self.post_attn_grn(attn_out)
            attn_processed = self.post_attn_norm(attn_processed + lstm_processed)

            # ── 5. Position-wise Feedforward ──────────────────────────────
            ff_out = self.ff_grn(attn_processed)

            # ── 6. Output (use last timestep) ────────────────────────────
            final_hidden = ff_out[:, -1, :]  # (batch, hidden)

            # Classification heads
            logits = []
            probs = []
            for head in self.classification_heads:
                l = head(final_hidden)
                logits.append(l)
                probs.append(F.softmax(l, dim=-1))

            # Quantile heads
            quantiles = []
            for head in self.quantile_heads:
                quantiles.append(head(final_hidden))

            # Uncertainty
            uncertainty = self.uncertainty_head(final_hidden)

            return {
                # Raw logits (for loss computation)
                "logits_5m": logits[0],
                "logits_15m": logits[1],
                "logits_1h": logits[2],
                # Probabilities (for inference)
                "probs_5m": probs[0],
                "probs_15m": probs[1],
                "probs_1h": probs[2],
                # Quantile forecasts
                "quantiles_5m": quantiles[0],
                "quantiles_15m": quantiles[1],
                "quantiles_1h": quantiles[2],
                # Uncertainty
                "uncertainty": uncertainty,
                # Interpretability
                "feature_weights": var_weights,
                "attention_weights": attn_weights,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Output Dataclass ────────────────────────────────────────────────────────

@dataclass
class HydraSignal:
    """Structured output from HYDRA inference."""
    # Primary signal
    direction: str          # "LONG", "SHORT", "FLAT"
    confidence: float       # 0-1 probability of predicted direction
    uncertainty: float      # Aleatoric uncertainty

    # Per-horizon probabilities
    probs_5m: list[float]   # [P(SHORT), P(FLAT), P(LONG)]
    probs_15m: list[float]
    probs_1h: list[float]

    # Quantile forecasts (expected return %)
    quantiles_5m: list[float]   # [P10, P50, P90]
    quantiles_15m: list[float]
    quantiles_1h: list[float]

    # Feature importance (top 10)
    top_features: list[tuple[str, float]] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if direction != FLAT and confidence > 0.55."""
        return self.direction != "FLAT" and self.confidence > 0.55

    @property
    def horizon_agreement(self) -> float:
        """Fraction of horizons that agree on direction (0-1)."""
        dirs = []
        for probs in [self.probs_5m, self.probs_15m, self.probs_1h]:
            idx = max(range(3), key=lambda i: probs[i])
            dirs.append(idx)
        # Count most common direction
        from collections import Counter
        most_common = Counter(dirs).most_common(1)[0][1]
        return most_common / 3.0
