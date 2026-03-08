"""
APHELION HYDRA — LSTM Sub-model
Bidirectional LSTM with self-attention for sequence momentum mapping.
Captures regime persistence over the 64-bar lookback.
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
    """HYDRA-LSTM configuration."""
    n_continuous: int = len(CONTINUOUS_FEATURES)
    n_categorical: int = len(CATEGORICAL_FEATURES)
    cat_embedding_dims: list[int] = (8, 8)
    cat_cardinalities: list[int] = (5, 7)
    
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    
    # Outputs matching TFT
    n_horizons: int = 3
    n_classes: int = 3


if HAS_TORCH:
    class SelfAttention(nn.Module):
        """Standard self-attention over sequence."""
        def __init__(self, hidden_size: int):
            super().__init__()
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.scale = math.sqrt(hidden_size)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq_len, hidden)
            Q = self.query(x)
            K = self.key(x)
            V = self.value(x)

            scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
            attn = F.softmax(scores, dim=-1)
            context = torch.bmm(attn, V)
            return context, attn


    class HydraLSTM(nn.Module):
        """
        Bidirectional LSTM + Attention model.
        Extracts sequential persistence features.
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

            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0,
                bidirectional=True,
            )

            # Bidirectional means output is 2 * hidden_size
            self.attention = SelfAttention(cfg.hidden_size * 2)
            
            self.dropout = nn.Dropout(cfg.dropout)
            self.fc_proj = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size)
            self.norm = nn.LayerNorm(cfg.hidden_size)

            # Auxiliary heads (to ensure latent rep captures directional info)
            self.aux_heads = nn.ModuleList([
                nn.Linear(cfg.hidden_size, cfg.n_classes)
                for _ in range(cfg.n_horizons)
            ])
            
            self._init_weights()

        def _init_weights(self):
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        def forward(
            self,
            cont_inputs: torch.Tensor,
            cat_inputs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """
            Args:
                cont_inputs: (batch, seq, n_cont)
                cat_inputs: (batch, seq, n_cat)
            """
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

            # LSTM Pass
            lstm_out, _ = self.lstm(x)  # (batch, seq, 2*hidden)

            # Attention
            attn_out, attn_weights = self.attention(lstm_out)
            
            # Using the last active timestep from attention output
            last_state = attn_out[:, -1, :]  # (batch, 2*hidden)
            
            # Projection to target dim
            proj = self.norm(self.fc_proj(self.dropout(last_state)))  # (batch, hidden)
            
            # Auxiliary classification
            aux_logits = [head(proj) for head in self.aux_heads]

            return {
                "latent": proj,              # Used by the Gate
                "aux_logits": aux_logits,    # [logits_5m, logits_15m, logits_1h]
                "attention_weights": attn_weights,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
