"""
APHELION HYDRA — Dynamic Attention Gate (Full Ensemble)
The master neural gate joining TFT, LSTM, CNN, and MoE.
Dynamically weights predictions from all sub-models based on market regime context.
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


@dataclass
class EnsembleConfig:
    """HYDRA Full Ensemble configuration."""
    tft_config: TFTConfig = field(default_factory=TFTConfig)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)
    cnn_config: CNNConfig = field(default_factory=CNNConfig)
    moe_config: MoEConfig = field(default_factory=MoEConfig)
    tcn_config: TCNConfig = field(default_factory=TCNConfig)
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)

    # Master projection size
    gate_hidden_size: int = 256
    dropout: float = 0.2
    n_horizons: int = 3
    n_classes: int = 3
    n_quantiles: int = 3


if HAS_TORCH:
    class HydraGate(nn.Module):
        """
        Master Dynamic Attention Gate.
        
        Sub-models:
        1. TFT (Baseline Multi-horizon logic & Interpretability)
        2. LSTM (Sequence Momentum / State persistence)
        3. CNN (Structural Pattern matching)
        4. MoE (Regime-specific specializations)
        5. TCN (Long-range temporal dependencies via dilated causal convolutions)
        6. Transformer (Global context extraction via multi-head self-attention)
        """
        def __init__(self, config: Optional[EnsembleConfig] = None):
            super().__init__()
            self.config = config or EnsembleConfig()
            cfg = self.config

            # Initialize all sub-models
            self.tft = TemporalFusionTransformer(cfg.tft_config)
            self.lstm = HydraLSTM(cfg.lstm_config)
            self.cnn = HydraCNN(cfg.cnn_config)
            self.moe = HydraMoE(cfg.moe_config)
            self.tcn = HydraTCN(cfg.tcn_config)
            self.transformer = HydraTransformer(cfg.transformer_config)

            # Projection from sub-model latent sizes -> gate hidden size
            self.lstm_proj = nn.Linear(cfg.lstm_config.hidden_size, cfg.gate_hidden_size)
            self.cnn_proj = nn.Linear(cfg.cnn_config.hidden_size, cfg.gate_hidden_size)
            self.moe_proj = nn.Linear(cfg.moe_config.hidden_size, cfg.gate_hidden_size)
            self.tcn_proj = nn.Linear(cfg.tcn_config.hidden_size, cfg.gate_hidden_size)
            self.transformer_proj = nn.Linear(cfg.transformer_config.hidden_size, cfg.gate_hidden_size)
            
            # The TFT doesn't output a single latent naturally in its public API
            # that we've exposed, but we can capture its final FF output.
            # To avoid refactoring TFT internals, we route all TFT 'logits' and context 
            # through a small adapter, or rely on the combined latent stream.
            
            # For this Gate, we'll design the Attention to combine the 3 new latents + TFT logits
            tft_info_size = (cfg.n_horizons * cfg.n_classes) + (cfg.n_horizons * cfg.n_quantiles) + 1 # +1 for uncertainty
            self.tft_adapter = nn.Sequential(
                nn.Linear(tft_info_size, cfg.gate_hidden_size),
                nn.ReLU(),
            )

            # Master Attention Query (Learned Parameter)
            self.master_query = nn.Parameter(torch.randn(1, 1, cfg.gate_hidden_size))

            # Master Attention mechanisms (Scaled Dot Product over the 4 model latents)
            self.attn_k = nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size)
            self.attn_v = nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size)
            
            # Final Heads (Classification, Quantiles, Uncertainty) on top of Gate
            self.classification_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.gate_hidden_size // 2, cfg.n_classes),
                )
                for _ in range(cfg.n_horizons)
            ])

            self.quantile_heads = nn.ModuleList([
                nn.Linear(cfg.gate_hidden_size, cfg.n_quantiles)
                for _ in range(cfg.n_horizons)
            ])

            self.uncertainty_head = nn.Sequential(
                nn.Linear(cfg.gate_hidden_size, cfg.gate_hidden_size // 4),
                nn.ReLU(),
                nn.Linear(cfg.gate_hidden_size // 4, 1),
                nn.Softplus(),
            )
            
            self._init_weights()

        def _init_weights(self):
            nn.init.xavier_uniform_(self.master_query)
            for name, param in self.named_parameters():
                if 'proj' in name or 'adapter' in name or 'head' in name or 'attn' in name:
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
            Ensemble forward pass.
            """
            # 1. Run Sub-Models
            tft_out = self.tft(cont_inputs, cat_inputs)
            lstm_out = self.lstm(cont_inputs, cat_inputs)
            cnn_out = self.cnn(cont_inputs, cat_inputs)
            moe_out = self.moe(cont_inputs, cat_inputs)

            # 2. Extract & Project Latents
            # Project LSTM, CNN, MoE to common dimension
            l_lstm = self.lstm_proj(lstm_out["latent"]).unsqueeze(1)  # (batch, 1, hidden)
            l_cnn = self.cnn_proj(cnn_out["latent"]).unsqueeze(1)     # (batch, 1, hidden)
            l_moe = self.moe_proj(moe_out["latent"]).unsqueeze(1)     # (batch, 1, hidden)
            
            # Flatten TFT outputs to create a TFT Latent
            tft_flat = torch.cat([
                tft_out["logits_5m"], tft_out["logits_15m"], tft_out["logits_1h"],
                tft_out["quantiles_5m"], tft_out["quantiles_15m"], tft_out["quantiles_1h"],
                tft_out["uncertainty"]
            ], dim=-1)
            l_tft = self.tft_adapter(tft_flat).unsqueeze(1)  # (batch, 1, hidden)

            # TCN and Transformer latents
            tcn_out = self.tcn(cont_inputs, cat_inputs)
            trans_out = self.transformer(cont_inputs, cat_inputs)
            l_tcn = self.tcn_proj(tcn_out["latent"]).unsqueeze(1)      # (batch, 1, hidden)
            l_trans = self.transformer_proj(trans_out["latent"]).unsqueeze(1)  # (batch, 1, hidden)

            # Stack into a sequence of 6 "tokens" representing the 6 DL models
            # Shape: (batch, 6, gate_hidden_size)
            latents = torch.cat([l_tft, l_lstm, l_cnn, l_moe, l_tcn, l_trans], dim=1)

            # 3. Dynamic Attention Gate
            batch_size = latents.size(0)
            
            # Q: (batch, 1, hidden)
            Q = self.master_query.expand(batch_size, -1, -1)
            
            # K, V: (batch, 6, hidden)
            K = self.attn_k(latents)
            V = self.attn_v(latents)
            
            # Attention scores: (batch, 1, 4)
            d_k = K.size(-1)
            scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Context vector (batch, 1, hidden) -> (batch, hidden)
            context = torch.bmm(attn_weights, V).squeeze(1)

            # 4. Master Output Heads
            logits = []
            probs = []
            for head in self.classification_heads:
                l = head(context)
                logits.append(l)
                probs.append(F.softmax(l, dim=-1))

            quantiles = []
            for head in self.quantile_heads:
                quantiles.append(head(context))

            uncertainty = self.uncertainty_head(context)

            # Consolidate all metrics to allow multi-loss optimization
            # `HydraTrainer` will use logits_5m, quantiles_5m, aux_lstm_logits, etc.
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
                
                # ENSEMBLE INSIGHTS
                # Attention across [TFT, LSTM, CNN, MoE]
                "gate_attention_weights": attn_weights, 
                "moe_routing_weights": moe_out["routing_weights"],
                
                # AUXILIARY LOSS LOGITS (For trainer.py)
                "tft_logits": [tft_out["logits_5m"], tft_out["logits_15m"], tft_out["logits_1h"]],
                "lstm_logits": lstm_out["aux_logits"],
                "cnn_logits": cnn_out["aux_logits"],
                "moe_logits": moe_out["aux_logits"],
                "tcn_logits": tcn_out["aux_logits"],
                "transformer_logits": trans_out["aux_logits"],

                # Interpretability forward-pass from TFT
                "feature_weights": tft_out["feature_weights"]
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
