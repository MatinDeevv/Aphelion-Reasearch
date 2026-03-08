"""
APHELION HYDRA — Mixture of Experts (MoE) Sub-model
Routes input dynamically to 4 specialists: TREND, RANGE, VOL_EXPANSION, NEWS_REACTION.
"""

from __future__ import annotations

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
class MoEConfig:
    """HYDRA-MoE configuration."""
    n_continuous: int = len(CONTINUOUS_FEATURES)
    n_categorical: int = len(CATEGORICAL_FEATURES)
    cat_embedding_dims: list[int] = (8, 8)
    cat_cardinalities: list[int] = (5, 7)
    
    hidden_size: int = 128
    num_experts: int = 4  # (TREND, RANGE, VOL_EXPANSION, NEWS_REACTION)
    dropout: float = 0.2
    
    # Outputs matching TFT
    n_horizons: int = 3
    n_classes: int = 3


if HAS_TORCH:
    class ExpertNetwork(nn.Module):
        """A specialist feedforward network."""
        def __init__(self, in_features: int, hidden_size: int, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class HydraMoE(nn.Module):
        """
        Mixture of Experts model.
        Uses a Gating Network (router) to dynamically weight the outputs of N expert networks.
        """
        def __init__(self, config: Optional[MoEConfig] = None):
            super().__init__()
            self.config = config or MoEConfig()
            cfg = self.config

            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, emb_dim)
                for card, emb_dim in zip(cfg.cat_cardinalities, cfg.cat_embedding_dims)
            ])

            # Use only the most recent bar (index -1) for MoE routing
            input_dim = cfg.n_continuous + sum(cfg.cat_embedding_dims)

            # 4 Specialist Experts
            self.experts = nn.ModuleList([
                ExpertNetwork(input_dim, cfg.hidden_size, cfg.dropout)
                for _ in range(cfg.num_experts)
            ])

            # Router / Gating Network
            self.router = nn.Sequential(
                nn.Linear(input_dim, cfg.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size // 2, cfg.num_experts)
            )
            
            # Post-MoE projection
            self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
            self.norm = nn.LayerNorm(cfg.hidden_size)
            self.dropout = nn.Dropout(cfg.dropout)
            self.elu = nn.ELU()
            
            # Auxiliary classification
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
            # MoE only looks at the CURRENT state (last bar in sequence)
            # to determine the immediate regime/pattern (Trend/Range/News)
            cont_last = cont_inputs[:, -1, :]  # (batch, n_cont)
            cat_last = cat_inputs[:, -1, :]    # (batch, n_cat)

            cat_embedded = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_embedded.append(emb(cat_last[:, i]))
            
            if cat_embedded:
                cat_repr = torch.cat(cat_embedded, dim=-1)
                x = torch.cat([cont_last, cat_repr], dim=-1)  # (batch, input_dim)
            else:
                x = cont_last

            # Router probabilities: (batch, num_experts)
            router_logits = self.router(x)
            routing_weights = F.softmax(router_logits, dim=-1)

            # Expert outputs: (batch, num_experts, hidden_size)
            expert_outputs = torch.stack([
                expert(x) for expert in self.experts
            ], dim=1)

            # Weighted combination: (batch, hidden_size)
            # (batch, 1, num_experts) @ (batch, num_experts, hidden) -> (batch, 1, hidden) -> (batch, hidden)
            combined_output = torch.bmm(
                routing_weights.unsqueeze(1), expert_outputs
            ).squeeze(1)
            
            proj = self.norm(self.elu(self.dropout(self.proj(combined_output))))

            # Auxiliary classification
            aux_logits = [head(proj) for head in self.aux_heads]

            return {
                "latent": proj,              # Used by the Gate
                "aux_logits": aux_logits,    # [logits_5m, logits_15m, logits_1h]
                "routing_weights": routing_weights,  # To track which expert is active
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
