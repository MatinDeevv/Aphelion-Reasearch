"""
APHELION HYDRA — CNN Sub-model
1D Convolutional ResNet over the 64-bar window.
Detects structural patterns (engulfing candles, ranges, breakouts) visually.
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
class CNNConfig:
    """HYDRA-CNN configuration."""
    n_continuous: int = len(CONTINUOUS_FEATURES)
    lookback: int = 64
    
    channels: list[int] = (32, 64, 128)
    kernel_sizes: list[int] = (3, 5, 7)
    dropout: float = 0.2
    
    hidden_size: int = 128
    
    # Outputs matching TFT
    n_horizons: int = 3
    n_classes: int = 3


if HAS_TORCH:
    class ResidualBlock1D(nn.Module):
        """1D ResNet block."""
        def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1):
            super().__init__()
            padding = kernel_size // 2
            self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride, padding, bias=False)
            self.bn1 = nn.BatchNorm1d(out_c)
            self.elu = nn.ELU()
            self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, 1, padding, bias=False)
            self.bn2 = nn.BatchNorm1d(out_c)
            self.dropout = nn.Dropout(0.2)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_c)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = self.shortcut(x)
            out = self.elu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out += res
            return self.elu(out)


    class HydraCNN(nn.Module):
        """
        Convolutional Pattern Recognition network.
        Processes the sequence like a 1D image to find geometric patterns.
        """
        def __init__(self, config: Optional[CNNConfig] = None):
            super().__init__()
            self.config = config or CNNConfig()
            cfg = self.config

            self.input_proj = nn.Conv1d(cfg.n_continuous, cfg.channels[0], kernel_size=1)
            
            blocks = []
            in_c = cfg.channels[0]
            for out_c, k in zip(cfg.channels, cfg.kernel_sizes):
                blocks.append(ResidualBlock1D(in_c, out_c, kernel_size=k, stride=2))
                in_c = out_c
                
            self.resnet = nn.Sequential(*blocks)
            
            # Global Average Pooling
            self.gap = nn.AdaptiveAvgPool1d(1)
            
            self.fc1 = nn.Linear(cfg.channels[-1], cfg.hidden_size)
            self.norm = nn.LayerNorm(cfg.hidden_size)
            self.dropout = nn.Dropout(cfg.dropout)
            self.elu = nn.ELU()
            
            self.aux_heads = nn.ModuleList([
                nn.Linear(cfg.hidden_size, cfg.n_classes)
                for _ in range(cfg.n_horizons)
            ])
            
            self._init_weights()

        def _init_weights(self):
            for name, param in self.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
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
                cat_inputs: Ignored by CNN (assumes categorical info isn't visual geometry)
            """
            # Conv1d expects (batch, channels, seq_len)
            x = cont_inputs.transpose(1, 2)
            
            x = self.input_proj(x)
            x = self.resnet(x)
            x = self.gap(x).squeeze(-1)  # (batch, out_c)
            
            proj = self.norm(self.elu(self.dropout(self.fc1(x))))  # (batch, hidden_size)
            
            # Auxiliary classification
            aux_logits = [head(proj) for head in self.aux_heads]

            return {
                "latent": proj,              # Used by the Gate
                "aux_logits": aux_logits,    # [logits_5m, logits_15m, logits_1h]
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
