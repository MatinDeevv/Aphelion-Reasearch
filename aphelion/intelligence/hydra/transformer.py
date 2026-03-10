"""
APHELION HYDRA — Transformer (SUPER INSANE Edition)
Deep Pre-LN Transformer with SwiGLU Feedforward, Rotary-style Learnable
Positional Encoding, and Multi-Scale Temporal Aggregation.
8 layers, 8 heads, 384 hidden — global context extraction at maximum power.
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


from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES


@dataclass
class TransformerConfig:
    input_size: int = len(CONTINUOUS_FEATURES)
    hidden_size: int = 384
    n_heads: int = 8
    n_layers: int = 8
    dim_feedforward: int = 1536  # 4x hidden for SwiGLU
    dropout: float = 0.15
    attention_dropout: float = 0.1
    n_classes: int = 3
    n_horizons: int = 3
    max_seq_len: int = 512
    stochastic_depth_rate: float = 0.1  # Layer drop probability


if HAS_TORCH:
    class LearnablePositionalEncoding(nn.Module):
        """Combined sinusoidal + learnable positional encoding."""

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            # Fixed sinusoidal
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            self.register_buffer("sinusoidal_pe", pe.unsqueeze(0))

            # Learnable offset
            self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.size(1)
            pe = self.sinusoidal_pe[:, :seq_len] + self.learnable_pe[:, :seq_len]
            return self.dropout(x + pe)

    class SwiGLUFeedForward(nn.Module):
        """SwiGLU activation — superior to ReLU/GELU for transformers."""

        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            # SwiGLU splits the output, so we need 2 * d_ff / 3 * 2 ≈ d_ff
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

    class PreNormTransformerLayer(nn.Module):
        """Pre-LN Transformer layer with SwiGLU and stochastic depth."""

        def __init__(self, d_model: int, n_heads: int, d_ff: int,
                     dropout: float = 0.1, attn_dropout: float = 0.1,
                     drop_path_rate: float = 0.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=attn_dropout, batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
            self.drop_path_rate = drop_path_rate

        def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
            """Stochastic depth — randomly skip layers during training."""
            if not self.training or self.drop_path_rate == 0.0:
                return x
            keep = torch.rand(1, device=x.device) > self.drop_path_rate
            return x * keep.float() / max(1.0 - self.drop_path_rate, 1e-6)

        def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
            # Pre-norm self-attention
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
            x = x + self._drop_path(attn_out)

            # Pre-norm SwiGLU feedforward
            normed = self.norm2(x)
            ff_out = self.ffn(normed)
            x = x + self._drop_path(ff_out)

            return x

    class HydraTransformer(nn.Module):
        """
        SUPER INSANE Transformer encoder.
        - 8 Pre-LN Transformer layers with SwiGLU activations
        - 8 attention heads with 384 hidden dimensions
        - Combined sinusoidal + learnable positional encoding
        - Stochastic depth for regularization
        - Multi-scale temporal aggregation (last + mean + CLS token)
        - Deep auxiliary prediction heads
        """

        def __init__(self, config: Optional[TransformerConfig] = None):
            super().__init__()
            self.config = config or TransformerConfig()
            cfg = self.config

            self.input_proj = nn.Sequential(
                nn.Linear(cfg.input_size, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.pos_encoder = LearnablePositionalEncoding(
                cfg.hidden_size, cfg.max_seq_len, cfg.dropout,
            )

            # Learnable CLS token for sequence summary
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_size) * 0.02)

            # Stochastic depth schedule (linear increase)
            drop_rates = [cfg.stochastic_depth_rate * i / max(cfg.n_layers - 1, 1)
                          for i in range(cfg.n_layers)]

            self.layers = nn.ModuleList([
                PreNormTransformerLayer(
                    cfg.hidden_size, cfg.n_heads, cfg.dim_feedforward,
                    cfg.dropout, cfg.attention_dropout, drop_rates[i],
                )
                for i in range(cfg.n_layers)
            ])

            self.final_norm = nn.LayerNorm(cfg.hidden_size)

            # Multi-scale aggregation: CLS + last + mean → project
            self.latent_proj = nn.Sequential(
                nn.Linear(cfg.hidden_size * 3, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(cfg.hidden_size),
            )

            # Deep auxiliary heads
            self.aux_head = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size // 2, cfg.n_classes * cfg.n_horizons),
            )
            self._n_classes = cfg.n_classes
            self._n_horizons = cfg.n_horizons

            # Causal mask
            self._register_causal_mask(cfg.max_seq_len + 1)  # +1 for CLS

        def _register_causal_mask(self, max_len: int):
            mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
            self.register_buffer("causal_mask", mask, persistent=False)

        def forward(self, cont_inputs: torch.Tensor,
                    cat_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            batch, seq_len, _ = cont_inputs.shape

            # Input projection
            x = self.input_proj(cont_inputs)
            x = self.pos_encoder(x)

            # Prepend CLS token
            cls = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (batch, 1+seq, hidden)

            # Causal mask
            total_len = x.size(1)
            mask = self.causal_mask[:total_len, :total_len].to(x.device)

            # Forward through transformer layers
            for layer in self.layers:
                x = layer(x, mask=mask)

            x = self.final_norm(x)

            # Multi-scale aggregation
            cls_out = x[:, 0, :]            # CLS token
            last_out = x[:, -1, :]          # Last timestep
            mean_out = x[:, 1:, :].mean(1)  # Mean of sequence (exclude CLS)

            combined = torch.cat([cls_out, last_out, mean_out], dim=-1)
            latent = self.latent_proj(combined)

            # Auxiliary heads
            aux = self.aux_head(latent)
            aux = aux.view(-1, self._n_horizons, self._n_classes)

            return {
                "latent": latent,
                "aux_logits": [aux[:, h, :] for h in range(self._n_horizons)],
                "aux_logits_all": aux,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
