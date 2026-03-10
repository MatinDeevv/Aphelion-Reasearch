"""
APHELION HYDRA — Mixture of Experts (SUPER INSANE Edition)
8 Specialist Experts with Top-K Noisy Gating, Load Balancing Loss,
Expert Dropout, and Deep Expert Networks.
Experts: TREND_BULL, TREND_BEAR, RANGE, VOL_EXPANSION, MEAN_REVERT,
         BREAKOUT, NEWS_SPIKE, REGIME_TRANSITION.
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
    """HYDRA-MoE configuration — SUPER INSANE defaults."""
    n_continuous: int = len(CONTINUOUS_FEATURES)
    n_categorical: int = len(CATEGORICAL_FEATURES)
    cat_embedding_dims: list[int] = (8, 8)
    cat_cardinalities: list[int] = (5, 7)

    hidden_size: int = 384
    expert_hidden_size: int = 512  # Internal expert width
    num_experts: int = 8
    top_k: int = 2  # Top-K routing — only activate K experts per sample
    dropout: float = 0.15
    expert_dropout: float = 0.1  # Randomly drop experts during training
    noise_std: float = 0.1  # Noisy gating for exploration
    load_balance_weight: float = 0.01  # Load balancing loss weight

    # Outputs matching TFT
    n_horizons: int = 3
    n_classes: int = 3


if HAS_TORCH:
    class DeepExpertNetwork(nn.Module):
        """Deep specialist network with residual and gating."""

        def __init__(self, in_features: int, hidden_size: int,
                     expert_hidden: int, dropout: float = 0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, expert_hidden),
                nn.LayerNorm(expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            # Skip connection
            self.skip = nn.Linear(in_features, hidden_size) if in_features != hidden_size else nn.Identity()
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skip = self.skip(x)
            out = self.net(x)
            g = self.gate(out)
            return g * out + (1 - g) * skip

    class NoisyTopKRouter(nn.Module):
        """
        Noisy top-K gating — adds learned noise for exploration,
        only activates top-K experts per sample for computational efficiency.
        Includes load balancing loss to prevent expert collapse.
        """

        def __init__(self, input_dim: int, num_experts: int, top_k: int = 2,
                     noise_std: float = 0.1, dropout: float = 0.15):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.noise_std = noise_std

            self.gate = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_experts),
            )
            self.noise_proj = nn.Linear(input_dim, num_experts)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns:
                routing_weights: (batch, num_experts) — sparse, only top_k are non-zero
                top_k_indices: (batch, top_k) — which experts are active
                load_balance_loss: scalar — auxiliary loss for balanced routing
            """
            logits = self.gate(x)

            # Add noise during training for exploration
            if self.training and self.noise_std > 0:
                noise = F.softplus(self.noise_proj(x))
                logits = logits + torch.randn_like(logits) * noise * self.noise_std

            # Top-K selection
            top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_vals.float(), dim=-1)

            # Create sparse routing weights (float32 to avoid AMP dtype mismatch)
            routing_weights = torch.zeros(logits.shape, device=logits.device, dtype=torch.float32)
            routing_weights.scatter_(1, top_k_idx, top_k_weights)

            # Load balancing loss: encourage uniform expert usage
            # f_i = fraction of tokens routed to expert i
            # P_i = mean probability assigned to expert i
            # loss = num_experts * sum(f_i * P_i)
            probs = F.softmax(logits.float(), dim=-1)
            expert_mask = torch.zeros(logits.shape, device=logits.device, dtype=torch.float32)
            expert_mask.scatter_(1, top_k_idx, torch.ones_like(top_k_weights))
            f = expert_mask.mean(dim=0)  # (num_experts,) fraction
            p = probs.mean(dim=0)        # (num_experts,) mean prob
            load_balance_loss = self.num_experts * (f * p).sum()

            return routing_weights, top_k_idx, load_balance_loss

    class HydraMoE(nn.Module):
        """
        SUPER INSANE Mixture of Experts.
        - 8 deep specialist experts with residual gating
        - Noisy top-2 routing for sparse expert activation
        - Expert dropout for regularization
        - Load balancing loss to prevent expert collapse
        - Sequence-aware routing (uses both last bar and sequence summary)
        """

        def __init__(self, config: Optional[MoEConfig] = None):
            super().__init__()
            self.config = config or MoEConfig()
            cfg = self.config

            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, emb_dim)
                for card, emb_dim in zip(cfg.cat_cardinalities, cfg.cat_embedding_dims)
            ])

            raw_input_dim = cfg.n_continuous + sum(cfg.cat_embedding_dims)

            # Sequence summarizer — don't just use last bar, summarize the whole window
            self.seq_summarizer = nn.Sequential(
                nn.Linear(raw_input_dim, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )

            # Router input = last bar features + sequence summary
            router_input_dim = raw_input_dim + cfg.hidden_size

            # 8 Deep Specialist Experts
            self.experts = nn.ModuleList([
                DeepExpertNetwork(
                    router_input_dim, cfg.hidden_size,
                    cfg.expert_hidden_size, cfg.dropout,
                )
                for _ in range(cfg.num_experts)
            ])

            # Noisy Top-K Router
            self.router = NoisyTopKRouter(
                router_input_dim, cfg.num_experts,
                cfg.top_k, cfg.noise_std, cfg.dropout,
            )

            # Expert dropout
            self.expert_dropout = nn.Dropout(cfg.expert_dropout)

            # Post-MoE projection
            self.output_proj = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.LayerNorm(cfg.hidden_size),
            )

            # Deep auxiliary heads
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

        def forward(
            self,
            cont_inputs: torch.Tensor,
            cat_inputs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            cfg = self.config

            # Last bar features
            cont_last = cont_inputs[:, -1, :]
            cat_last = cat_inputs[:, -1, :]

            cat_embedded = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_embedded.append(emb(cat_last[:, i]))

            if cat_embedded:
                cat_repr = torch.cat(cat_embedded, dim=-1)
                last_bar = torch.cat([cont_last, cat_repr], dim=-1)
            else:
                last_bar = cont_last

            # Sequence summary (mean over sequence)
            seq_mean = cont_inputs.mean(dim=1)
            cat_mean_embedded = []
            for i, emb in enumerate(self.cat_embeddings):
                # Use mode of categorical (approximate with last)
                cat_mean_embedded.append(emb(cat_inputs[:, -1, i]))
            if cat_mean_embedded:
                cat_mean_repr = torch.cat(cat_mean_embedded, dim=-1)
                seq_input = torch.cat([seq_mean, cat_mean_repr], dim=-1)
            else:
                seq_input = seq_mean

            seq_summary = self.seq_summarizer(seq_input)

            # Router input combines last bar + sequence context
            router_input = torch.cat([last_bar, seq_summary], dim=-1)

            # Top-K routing
            routing_weights, top_k_idx, load_balance_loss = self.router(router_input)

            # Run ALL experts (for simplicity — in practice only top-k would run)
            expert_outputs = torch.stack([
                expert(router_input) for expert in self.experts
            ], dim=1)  # (batch, num_experts, hidden)

            # Apply expert dropout during training
            if self.training:
                expert_outputs = self.expert_dropout(expert_outputs)

            # Weighted combination using sparse routing weights
            # Cast routing_weights to same dtype as expert_outputs for bmm
            combined = torch.bmm(
                routing_weights.unsqueeze(1).to(expert_outputs.dtype), expert_outputs,
            ).squeeze(1)  # (batch, hidden)

            proj = self.output_proj(combined)

            # Auxiliary classification
            aux_logits = [head(proj) for head in self.aux_heads]

            return {
                "latent": proj,
                "aux_logits": aux_logits,
                "routing_weights": routing_weights,
                "load_balance_loss": load_balance_loss,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
