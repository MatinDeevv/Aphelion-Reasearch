"""
APHELION HYDRA — LSTM Sub-model (SUPER INSANE Edition)
Deep Bidirectional LSTM with Multi-Head Self-Attention, Highway Gating,
Layer Normalization, and Residual Connections.
Captures regime persistence and momentum dynamics across the full lookback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES


@dataclass
class LSTMConfig:
    """HYDRA-LSTM configuration — SUPER INSANE defaults."""
    n_continuous: int = len(CONTINUOUS_FEATURES)
    n_categorical: int = len(CATEGORICAL_FEATURES)
    cat_embedding_dims: list[int] = (8, 8)
    cat_cardinalities: list[int] = (5, 7)

    hidden_size: int = 384
    num_layers: int = 4
    n_attention_heads: int = 8
    dropout: float = 0.15
    attention_dropout: float = 0.1

    # Outputs matching TFT
    n_horizons: int = 3
    n_classes: int = 3


if HAS_TORCH:
    class MultiHeadSelfAttention(nn.Module):
        """Multi-head self-attention with causal masking and residual."""

        def __init__(self, hidden_size: int, n_heads: int = 8, dropout: float = 0.1,
                     max_seq_len: int = 128):
            super().__init__()
            assert hidden_size % n_heads == 0
            self.n_heads = n_heads
            self.d_k = hidden_size // n_heads
            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)
            self.W_v = nn.Linear(hidden_size, hidden_size)
            self.W_o = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size)

            # OPTIMIZED: Pre-register causal mask buffer — no allocation per forward call
            causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
            self.register_buffer("causal_mask", causal, persistent=False)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch, seq_len, _ = x.shape
            residual = x
            x = self.norm(x)

            Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

            # Use pre-buffered causal mask (no allocation)
            scores = scores.masked_fill(
                self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            context = torch.matmul(attn, V)
            context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            out = self.W_o(context)

            # Average attention across heads for interpretability
            avg_attn = attn.mean(dim=1)

            return residual + out, avg_attn

    class HighwayGate(nn.Module):
        """Highway gating for controlled information flow."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )
            self.transform = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            g = self.gate(x)
            t = self.transform(x)
            return g * t + (1 - g) * x

    class TemporalPooling(nn.Module):
        """Multi-scale temporal pooling — captures short, medium, long patterns."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.short_pool = nn.AdaptiveAvgPool1d(1)   # Global
            self.gate = nn.Linear(hidden_size * 3, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq, hidden)
            batch, seq_len, hidden = x.shape

            # Last state (most recent)
            last = x[:, -1, :]

            # Global average
            global_avg = x.mean(dim=1)

            # Global max
            global_max = x.max(dim=1).values

            # Combine
            combined = torch.cat([last, global_avg, global_max], dim=-1)
            return self.gate(combined)

    class HydraLSTM(nn.Module):
        """
        SUPER INSANE Bidirectional LSTM.
        - 4-layer BiLSTM with residual connections and layer norm
        - Multi-head self-attention (8 heads)
        - Highway gating for information flow control
        - Multi-scale temporal pooling
        - Deep auxiliary heads with residual
        """

        def __init__(self, config: Optional[LSTMConfig] = None):
            super().__init__()
            self.config = config or LSTMConfig()
            cfg = self.config

            # Categorical embeddings
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, emb_dim)
                for card, emb_dim in zip(cfg.cat_cardinalities, cfg.cat_embedding_dims)
            ])

            input_dim = cfg.n_continuous + sum(cfg.cat_embedding_dims)

            # Input projection with GELU
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )

            # Stacked BiLSTM layers with residual connections and LayerNorm
            self.lstm_layers = nn.ModuleList()
            self.lstm_norms = nn.ModuleList()
            self.lstm_dropouts = nn.ModuleList()
            self.lstm_projs = nn.ModuleList()

            for i in range(cfg.num_layers):
                layer_input = cfg.hidden_size
                self.lstm_layers.append(nn.LSTM(
                    input_size=layer_input,
                    hidden_size=cfg.hidden_size // 2,  # Half because bidirectional doubles it
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                ))
                self.lstm_norms.append(nn.LayerNorm(cfg.hidden_size))
                self.lstm_dropouts.append(nn.Dropout(cfg.dropout))
                # Skip projection if dims match (they always do in this config)
                self.lstm_projs.append(nn.Identity())

            # Multi-head self-attention over final LSTM output
            self.attention = MultiHeadSelfAttention(
                cfg.hidden_size, cfg.n_attention_heads, cfg.attention_dropout,
            )

            # Highway gating
            self.highway = HighwayGate(cfg.hidden_size)

            # Multi-scale temporal pooling
            self.temporal_pool = TemporalPooling(cfg.hidden_size)

            # Final projection
            self.output_proj = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(cfg.hidden_size),
            )

            # Deep auxiliary heads with residual
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_size // 2, cfg.n_classes),
                )
                for _ in range(cfg.n_horizons)
            ])

            self._init_weights()

        def _init_weights(self):
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            # LSTM forget-gate bias = 1.0 for better gradient flow
            for lstm in self.lstm_layers:
                for name, param in lstm.named_parameters():
                    if 'bias' in name:
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

        def forward(
            self,
            cont_inputs: torch.Tensor,
            cat_inputs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            cfg = self.config

            # Embed categorical
            cat_embedded = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_embedded.append(emb(cat_inputs[:, :, i]))

            if cat_embedded:
                cat_repr = torch.cat(cat_embedded, dim=-1)
                x = torch.cat([cont_inputs, cat_repr], dim=-1)
            else:
                x = cont_inputs

            # Input projection
            x = self.input_proj(x)  # (batch, seq, hidden)

            # Stacked BiLSTM with residual + LayerNorm
            for lstm, norm, drop, proj in zip(
                self.lstm_layers, self.lstm_norms, self.lstm_dropouts, self.lstm_projs
            ):
                residual = proj(x)
                lstm_out, _ = lstm(x)
                x = norm(drop(lstm_out) + residual)

            # Multi-head self-attention
            x, attn_weights = self.attention(x)

            # Highway gating
            x = self.highway(x)

            # Multi-scale temporal pooling
            pooled = self.temporal_pool(x)  # (batch, hidden)

            # Final projection
            proj = self.output_proj(pooled)

            # Auxiliary classification
            aux_logits = [head(proj) for head in self.aux_heads]

            return {
                "latent": proj,
                "aux_logits": aux_logits,
                "attention_weights": attn_weights,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
