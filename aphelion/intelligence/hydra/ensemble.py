"""
APHELION HYDRA — Dynamic Attention Gate (SUPER INSANE Edition)
The master neural gate joining all 6 sub-models via multi-head cross-attention,
cross-model interaction layers, stochastic model dropout, regime-aware gating,
and deep residual output heads.
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

from aphelion.intelligence.hydra.tft import TFTConfig, TemporalFusionTransformer
from aphelion.intelligence.hydra.lstm import LSTMConfig, HydraLSTM
from aphelion.intelligence.hydra.cnn import CNNConfig, HydraCNN
from aphelion.intelligence.hydra.moe import MoEConfig, HydraMoE
from aphelion.intelligence.hydra.tcn import TCNConfig, HydraTCN
from aphelion.intelligence.hydra.transformer import TransformerConfig, HydraTransformer


NUM_SUB_MODELS = 6  # TFT, LSTM, CNN, MoE, TCN, Transformer


@dataclass
class EnsembleConfig:
    """HYDRA Full Ensemble — SUPER INSANE configuration."""
    tft_config: TFTConfig = field(default_factory=TFTConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    cnn_config: CNNConfig = field(default_factory=CNNConfig)
    moe_config: MoEConfig = field(default_factory=MoEConfig)
    tcn_config: TCNConfig = field(default_factory=TCNConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)

    # SUPER INSANE Master gate
    gate_hidden_size: int = 512
    gate_n_heads: int = 8       # Multi-head cross-attention
    gate_n_interaction_layers: int = 2  # Cross-model interaction
    model_dropout: float = 0.1  # Stochastic model dropout
    dropout: float = 0.15
    n_horizons: int = 3
    n_classes: int = 3
    n_quantiles: int = 3


if HAS_TORCH:

    class CrossModelInteraction(nn.Module):
        """Transformer layer over sub-model latents for cross-model reasoning."""

        def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pre-norm self-attention
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + attn_out
            # Pre-norm FFN
            normed = self.norm2(x)
            x = x + self.ffn(normed)
            return x

    class MultiHeadCrossAttentionGate(nn.Module):
        """Multi-head cross-attention from learnable queries to sub-model latents."""

        def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            assert d_model % n_heads == 0

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, query: torch.Tensor,
                    kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            batch = query.size(0)
            n_queries = query.size(1)
            n_kv = kv.size(1)

            Q = self.W_q(query).view(batch, n_queries, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(kv).view(batch, n_kv, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(kv).view(batch, n_kv, self.n_heads, self.d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch, n_queries, -1)
            context = self.W_o(context)
            context = self.norm(context)

            return context, attn_weights

    class DeepResidualHead(nn.Module):
        """Deep output head with residual connection."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     dropout: float = 0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
            self.skip = nn.Linear(input_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x) + self.skip(x)


    class HydraGate(nn.Module):
        """
        SUPER INSANE Master Dynamic Attention Gate.

        Architecture:
        1. Run all 6 sub-models in parallel
        2. Project all latents to common gate dimension (512)
        3. Cross-model interaction layers (2x self-attention over 6 model tokens)
        4. Stochastic model dropout (randomly zero models during training)
        5. Multi-head cross-attention gate (8 heads) — learnable queries attend to models
        6. Deep residual output heads (classification, quantile, uncertainty, confidence)
        7. MoE load balance loss passthrough
        """

        def __init__(self, config: Optional[EnsembleConfig] = None):
            super().__init__()
            self.config = config or EnsembleConfig()
            cfg = self.config

            # ── Sub-Models ────────────────────────────────────────────────
            self.tft = TemporalFusionTransformer(cfg.tft_config)
            self.lstm = HydraLSTM(cfg.lstm_config)
            self.cnn = HydraCNN(cfg.cnn_config)
            self.moe = HydraMoE(cfg.moe_config)
            self.tcn = HydraTCN(cfg.tcn_config)
            self.transformer = HydraTransformer(cfg.transformer_config)

            # ── Latent Projections → Gate Hidden ──────────────────────────
            self.lstm_proj = nn.Sequential(
                nn.Linear(cfg.lstm_config.hidden_size, cfg.gate_hidden_size),
                nn.GELU(),
            )
            self.cnn_proj = nn.Sequential(
                nn.Linear(cfg.cnn_config.hidden_size, cfg.gate_hidden_size),
                nn.GELU(),
            )
            self.moe_proj = nn.Sequential(
                nn.Linear(cfg.moe_config.hidden_size, cfg.gate_hidden_size),
                nn.GELU(),
            )
            self.tcn_proj = nn.Sequential(
                nn.Linear(cfg.tcn_config.hidden_size, cfg.gate_hidden_size),
                nn.GELU(),
            )
            self.transformer_proj = nn.Sequential(
                nn.Linear(cfg.transformer_config.hidden_size, cfg.gate_hidden_size),
                nn.GELU(),
            )

            # TFT adapter
            tft_info_size = (
                cfg.n_horizons * cfg.n_classes +
                cfg.n_horizons * cfg.n_quantiles + 1
            )
            self.tft_adapter = nn.Sequential(
                nn.Linear(tft_info_size, cfg.gate_hidden_size),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size),
                nn.GELU(),
            )

            # ── Cross-Model Interaction Layers ────────────────────────────
            self.interaction_layers = nn.ModuleList([
                CrossModelInteraction(cfg.gate_hidden_size, cfg.gate_n_heads, cfg.dropout)
                for _ in range(cfg.gate_n_interaction_layers)
            ])
            self.interaction_norm = nn.LayerNorm(cfg.gate_hidden_size)

            # ── Multi-Head Cross-Attention Gate ───────────────────────────
            self.master_queries = nn.Parameter(
                torch.randn(1, cfg.n_horizons + 1, cfg.gate_hidden_size) * 0.02,
            )
            self.cross_attention = MultiHeadCrossAttentionGate(
                cfg.gate_hidden_size, cfg.gate_n_heads, cfg.dropout,
            )

            # ── Stochastic Model Dropout ──────────────────────────────────
            self.model_dropout_p = cfg.model_dropout

            # ── Deep Residual Output Heads ────────────────────────────────
            self.classification_heads = nn.ModuleList([
                DeepResidualHead(
                    cfg.gate_hidden_size, cfg.gate_hidden_size // 2,
                    cfg.n_classes, cfg.dropout,
                )
                for _ in range(cfg.n_horizons)
            ])

            self.quantile_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(cfg.gate_hidden_size // 4, cfg.n_quantiles),
                )
                for _ in range(cfg.n_horizons)
            ])

            self.uncertainty_head = nn.Sequential(
                nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size // 4),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.gate_hidden_size // 4, 1),
                nn.Softplus(),
            )

            self.confidence_head = nn.Sequential(
                nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size // 4),
                nn.GELU(),
                nn.Linear(cfg.gate_hidden_size // 4, 1),
                nn.Sigmoid(),
            )

            self._init_weights()

        def _init_weights(self):
            for name, param in self.named_parameters():
                if any(k in name for k in ('proj', 'adapter', 'head', 'attn',
                                            'interaction', 'confidence', 'master')):
                    if 'weight' in name and param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        def forward(
            self,
            cont_inputs: torch.Tensor,
            cat_inputs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            """SUPER INSANE Ensemble forward pass."""
            cfg = self.config

            # ── 1. Run All Sub-Models ─────────────────────────────────────
            tft_out = self.tft(cont_inputs, cat_inputs)
            lstm_out = self.lstm(cont_inputs, cat_inputs)
            cnn_out = self.cnn(cont_inputs, cat_inputs)
            moe_out = self.moe(cont_inputs, cat_inputs)
            tcn_out = self.tcn(cont_inputs, cat_inputs)
            trans_out = self.transformer(cont_inputs, cat_inputs)

            # ── 2. Project Latents ────────────────────────────────────────
            l_lstm = self.lstm_proj(lstm_out["latent"]).unsqueeze(1)
            l_cnn = self.cnn_proj(cnn_out["latent"]).unsqueeze(1)
            l_moe = self.moe_proj(moe_out["latent"]).unsqueeze(1)
            l_tcn = self.tcn_proj(tcn_out["latent"]).unsqueeze(1)
            l_trans = self.transformer_proj(trans_out["latent"]).unsqueeze(1)

            tft_flat = torch.cat([
                tft_out["logits_5m"], tft_out["logits_15m"], tft_out["logits_1h"],
                tft_out["quantiles_5m"], tft_out["quantiles_15m"], tft_out["quantiles_1h"],
                tft_out["uncertainty"],
            ], dim=-1)
            l_tft = self.tft_adapter(tft_flat).unsqueeze(1)

            # (batch, 6, gate_hidden_size)
            latents = torch.cat([l_tft, l_lstm, l_cnn, l_moe, l_tcn, l_trans], dim=1)

            # ── 3. Stochastic Model Dropout ───────────────────────────────
            if self.training and self.model_dropout_p > 0:
                # OPTIMIZED: Use deterministic mask generation — no while loop
                mask = (torch.rand(latents.size(0), NUM_SUB_MODELS, 1,
                                   device=latents.device) > self.model_dropout_p).float()
                # Ensure at least 2 models active — clamp to guarantee
                active = mask.sum(dim=1, keepdim=True)
                too_few = (active < 2.0).expand_as(mask)
                mask = torch.where(too_few, torch.ones_like(mask), mask)
                latents = latents * mask / max(1.0 - self.model_dropout_p, 0.5)

            # ── 4. Cross-Model Interaction ────────────────────────────────
            for layer in self.interaction_layers:
                latents = layer(latents)
            latents = self.interaction_norm(latents)

            # ── 5. Multi-Head Cross-Attention Gate ────────────────────────
            batch_size = latents.size(0)
            queries = self.master_queries.expand(batch_size, -1, -1)
            context, attn_weights = self.cross_attention(queries, latents)

            # Split: first n_horizons for per-horizon, last for global
            horizon_contexts = [context[:, i, :] for i in range(cfg.n_horizons)]
            global_context = context[:, -1, :]

            # ── 6. Output Heads ───────────────────────────────────────────
            logits = []
            probs = []
            for i, head in enumerate(self.classification_heads):
                l = head(horizon_contexts[i])
                logits.append(l)
                probs.append(F.softmax(l, dim=-1))

            quantiles = []
            for i, head in enumerate(self.quantile_heads):
                quantiles.append(head(horizon_contexts[i]))

            uncertainty = self.uncertainty_head(global_context)
            confidence = self.confidence_head(global_context)

            # MoE load balance loss
            moe_lb_loss = moe_out.get("load_balance_loss", torch.tensor(0.0))

            # Attention weights for interpretability: (batch, n_queries, n_models)
            gate_attn = attn_weights.mean(dim=1)

            return {
                # MASTER OUTPUTS
                "logits_5m": logits[0],
                "logits_15m": logits[1],
                "logits_1h": logits[2],
                "probs_5m": probs[0],
                "probs_15m": probs[1],
                "probs_1h": probs[2],
                "quantiles_5m": quantiles[0],
                "quantiles_15m": quantiles[1],
                "quantiles_1h": quantiles[2],
                "uncertainty": uncertainty,
                "confidence": confidence,

                # ENSEMBLE INSIGHTS
                "gate_attention_weights": gate_attn,
                "moe_routing_weights": moe_out["routing_weights"],
                "moe_load_balance_loss": moe_lb_loss,

                # AUXILIARY LOSS LOGITS
                "tft_logits": [tft_out["logits_5m"], tft_out["logits_15m"], tft_out["logits_1h"]],
                "lstm_logits": lstm_out["aux_logits"],
                "cnn_logits": cnn_out["aux_logits"],
                "moe_logits": moe_out["aux_logits"],
                "tcn_logits": tcn_out["aux_logits"],
                "transformer_logits": trans_out["aux_logits"],

                # Interpretability
                "feature_weights": tft_out["feature_weights"],
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
