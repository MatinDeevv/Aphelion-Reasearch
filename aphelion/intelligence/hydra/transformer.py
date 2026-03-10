"""
APHELION HYDRA — Vanilla Multi-Head Attention Transformer (Phase 7 v2)
Global context extraction for multi-horizon prediction.
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
    hidden_size: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.2
    n_classes: int = 3
    n_horizons: int = 3
    max_seq_len: int = 256


if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class HydraTransformer(nn.Module):
        """Vanilla Transformer encoder for HYDRA ensemble."""

        def __init__(self, config: Optional[TransformerConfig] = None):
            super().__init__()
            self.config = config or TransformerConfig()
            cfg = self.config

            self.input_proj = nn.Linear(cfg.input_size, cfg.hidden_size)
            self.pos_encoder = PositionalEncoding(cfg.hidden_size, cfg.max_seq_len, cfg.dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

            self.latent_proj = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )

            self.aux_head = nn.Linear(cfg.hidden_size, cfg.n_classes * cfg.n_horizons)
            self._n_classes = cfg.n_classes
            self._n_horizons = cfg.n_horizons

            # Causal mask
            self._register_causal_mask(cfg.max_seq_len)

        def _register_causal_mask(self, max_len: int):
            mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
            self.register_buffer("causal_mask", mask, persistent=False)

        def forward(self, cont_inputs: torch.Tensor,
                    cat_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            # cont_inputs: (batch, seq, features)
            seq_len = cont_inputs.size(1)
            x = self.input_proj(cont_inputs)
            x = self.pos_encoder(x)

            mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
            x = self.encoder(x, mask=mask)

            # Use last time-step as summary
            latent = self.latent_proj(x[:, -1, :])

            aux = self.aux_head(latent)
            aux = aux.view(-1, self._n_horizons, self._n_classes)

            return {
                "latent": latent,
                "aux_logits": aux[:, 0, :],
                "aux_logits_all": aux,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
