"""
APHELION HYDRA — CNN Sub-model (SUPER INSANE Edition)
Deep 1D ConvNet with Squeeze-and-Excitation, Inception-style Multi-Scale
Convolutions, Pre-activation Residual Blocks, and Channel Attention.
Detects structural patterns at multiple temporal resolutions.
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
    """HYDRA-CNN configuration — SUPER INSANE defaults."""
    n_continuous: int = len(CONTINUOUS_FEATURES)
    lookback: int = 64

    channels: list[int] = (64, 128, 256, 512)
    kernel_sizes: list[int] = (3, 5, 7, 9)
    dropout: float = 0.15
    se_reduction: int = 16  # Squeeze-and-Excitation reduction ratio

    hidden_size: int = 384

    # Outputs matching TFT
    n_horizons: int = 3
    n_classes: int = 3


if HAS_TORCH:
    class SqueezeExcitation1D(nn.Module):
        """Channel attention via Squeeze-and-Excitation."""

        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            mid = max(channels // reduction, 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels, mid),
                nn.GELU(),
                nn.Linear(mid, channels),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, channels, seq)
            scale = self.se(x).unsqueeze(-1)
            return x * scale

    class InceptionBlock1D(nn.Module):
        """Multi-scale parallel convolutions (inception-style)."""

        def __init__(self, in_c: int, out_c: int, dropout: float = 0.15):
            super().__init__()
            branch_c = out_c // 4

            # 1x1 (point-wise)
            self.branch1 = nn.Sequential(
                nn.Conv1d(in_c, branch_c, 1, bias=False),
                nn.BatchNorm1d(branch_c),
                nn.GELU(),
            )
            # 3x3 (short patterns)
            self.branch3 = nn.Sequential(
                nn.Conv1d(in_c, branch_c, 3, padding=1, bias=False),
                nn.BatchNorm1d(branch_c),
                nn.GELU(),
            )
            # 5x5 (medium patterns)
            self.branch5 = nn.Sequential(
                nn.Conv1d(in_c, branch_c, 5, padding=2, bias=False),
                nn.BatchNorm1d(branch_c),
                nn.GELU(),
            )
            # 7x7 (long patterns)
            self.branch7 = nn.Sequential(
                nn.Conv1d(in_c, out_c - 3 * branch_c, 7, padding=3, bias=False),
                nn.BatchNorm1d(out_c - 3 * branch_c),
                nn.GELU(),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b1 = self.branch1(x)
            b3 = self.branch3(x)
            b5 = self.branch5(x)
            b7 = self.branch7(x)
            return self.dropout(torch.cat([b1, b3, b5, b7], dim=1))

    class PreActResBlock1D(nn.Module):
        """Pre-activation ResNet block with SE attention."""

        def __init__(self, in_c: int, out_c: int, kernel_size: int = 3,
                     stride: int = 1, se_reduction: int = 16, dropout: float = 0.15):
            super().__init__()
            padding = kernel_size // 2
            self.bn1 = nn.BatchNorm1d(in_c)
            self.conv1 = nn.Conv1d(in_c, out_c, kernel_size, stride, padding, bias=False)
            self.bn2 = nn.BatchNorm1d(out_c)
            self.conv2 = nn.Conv1d(out_c, out_c, kernel_size, 1, padding, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.se = SqueezeExcitation1D(out_c, se_reduction)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                    nn.BatchNorm1d(out_c),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.shortcut(x)
            out = F.gelu(self.bn1(x))
            out = self.conv1(out)
            out = F.gelu(self.bn2(out))
            out = self.dropout(out)
            out = self.conv2(out)
            out = self.se(out)
            return out + residual

    class HydraCNN(nn.Module):
        """
        SUPER INSANE Convolutional Pattern Recognition network.
        - Inception-style multi-scale input processing
        - Pre-activation ResNet backbone with SE attention
        - Multi-scale feature aggregation
        - Deeper auxiliary heads
        """

        def __init__(self, config: Optional[CNNConfig] = None):
            super().__init__()
            self.config = config or CNNConfig()
            cfg = self.config

            # Multi-scale inception front-end
            self.inception = InceptionBlock1D(cfg.n_continuous, cfg.channels[0], cfg.dropout)

            # Deep ResNet backbone with SE blocks
            blocks = []
            in_c = cfg.channels[0]
            for out_c, k in zip(cfg.channels, cfg.kernel_sizes):
                blocks.append(PreActResBlock1D(
                    in_c, out_c, kernel_size=k, stride=2,
                    se_reduction=cfg.se_reduction, dropout=cfg.dropout,
                ))
                # Add a second block at each scale for depth
                blocks.append(PreActResBlock1D(
                    out_c, out_c, kernel_size=k, stride=1,
                    se_reduction=cfg.se_reduction, dropout=cfg.dropout,
                ))
                in_c = out_c

            self.resnet = nn.Sequential(*blocks)

            # Multi-scale pooling: both global average and global max
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.gmp = nn.AdaptiveMaxPool1d(1)

            # Projection from concatenated pooled features
            self.fc = nn.Sequential(
                nn.Linear(cfg.channels[-1] * 2, cfg.hidden_size),
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
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.zeros_(param)

        def forward(
            self,
            cont_inputs: torch.Tensor,
            cat_inputs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            # Conv1d expects (batch, channels, seq_len)
            x = cont_inputs.transpose(1, 2)

            # Multi-scale inception front-end
            x = self.inception(x)

            # Deep ResNet backbone
            x = self.resnet(x)

            # Dual pooling
            avg_pool = self.gap(x).squeeze(-1)
            max_pool = self.gmp(x).squeeze(-1)
            pooled = torch.cat([avg_pool, max_pool], dim=-1)

            # Project
            proj = self.fc(pooled)

            # Auxiliary classification
            aux_logits = [head(proj) for head in self.aux_heads]

            return {
                "latent": proj,
                "aux_logits": aux_logits,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
