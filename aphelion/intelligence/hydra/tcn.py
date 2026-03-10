"""
APHELION HYDRA — Temporal Convolutional Network (SUPER INSANE Edition)
Deep dilated causal convolutions with Gated Activations, Weight Normalization,
Dense Skip Connections, and Channel Attention.
Massive receptive field for long-range temporal dependency capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class TCNConfig:
    input_size: int = len(CONTINUOUS_FEATURES)
    hidden_size: int = 384
    num_channels: list[int] = None  # Channel sizes per layer
    kernel_size: int = 3
    dropout: float = 0.15
    n_classes: int = 3
    n_horizons: int = 3
    sequence_length: int = 512

    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [64, 128, 128, 256, 256, 512]


if HAS_TORCH:
    class GatedCausalConv1d(nn.Module):
        """Gated causal convolution — learned gating controls information flow."""

        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, dilation: int, dropout: float = 0.15):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            # Signal path
            self.conv_signal = nn.utils.parametrizations.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            )
            # Gate path
            self.conv_gate = nn.utils.parametrizations.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            )
            self.bn = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_padded = F.pad(x, (self.padding, 0))
            signal = torch.tanh(self.conv_signal(x_padded))
            gate = torch.sigmoid(self.conv_gate(x_padded))
            out = signal * gate  # Gated activation
            out = self.bn(out)
            out = self.dropout(out)
            return out

    class DenseTemporalBlock(nn.Module):
        """Residual block with gated causal convolutions and skip connection output."""

        def __init__(self, in_ch: int, out_ch: int,
                     kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.conv1 = GatedCausalConv1d(in_ch, out_ch, kernel_size, dilation, dropout)
            self.conv2 = GatedCausalConv1d(out_ch, out_ch, kernel_size, dilation, dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

            # 1x1 skip connection projection
            self.skip_proj = nn.Conv1d(out_ch, out_ch, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            residual = x if self.downsample is None else self.downsample(x)
            out = self.conv1(x)
            out = self.conv2(out)
            skip = self.skip_proj(out)
            return F.gelu(out + residual), skip

    class ChannelAttention1D(nn.Module):
        """Channel attention for temporal features."""

        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            mid = max(channels // reduction, 8)
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels * 2, mid),
                nn.GELU(),
                nn.Linear(mid, channels),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avg = self.avg_pool(x).squeeze(-1)
            mx = self.max_pool(x).squeeze(-1)
            scale = self.fc(torch.cat([avg, mx], dim=-1)).unsqueeze(-1)
            return x * scale

    class HydraTCN(nn.Module):
        """
        SUPER INSANE Temporal Convolutional Network.
        - 6-layer deep TCN with exponentially growing receptive field
        - Gated activations (tanh * sigmoid) for controlled information flow
        - Weight normalization for training stability
        - Dense skip connections aggregated for final prediction
        - Channel attention for feature refinement
        - Multi-scale temporal pooling
        """

        def __init__(self, config: Optional[TCNConfig] = None):
            super().__init__()
            self.config = config or TCNConfig()
            cfg = self.config

            self.input_proj = nn.Linear(cfg.input_size, cfg.num_channels[0])

            # Build layers with exponentially increasing dilation
            self.layers = nn.ModuleList()
            channels = cfg.num_channels
            for i in range(len(channels)):
                in_ch = channels[i - 1] if i > 0 else channels[0]
                out_ch = channels[i]
                dilation = 2 ** i
                self.layers.append(DenseTemporalBlock(
                    in_ch, out_ch, cfg.kernel_size, dilation, cfg.dropout,
                ))

            # Aggregate all skip connections
            total_skip = sum(channels)
            self.skip_aggregate = nn.Sequential(
                nn.Conv1d(total_skip, channels[-1], 1),
                nn.GELU(),
                nn.BatchNorm1d(channels[-1]),
            )

            # Channel attention
            self.channel_attn = ChannelAttention1D(channels[-1])

            # Multi-scale temporal pooling
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

            # Final projection
            self.latent_proj = nn.Sequential(
                nn.Linear(channels[-1] * 2, cfg.hidden_size),
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

        def forward(self, cont_inputs: torch.Tensor,
                    cat_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            x = self.input_proj(cont_inputs)
            x = x.transpose(1, 2)  # (batch, channels, seq)

            # Forward through layers, collecting skip connections
            skips = []
            for layer in self.layers:
                x, skip = layer(x)
                skips.append(skip)

            # Align skip connection lengths to the shortest
            min_len = min(s.size(-1) for s in skips)
            skips_aligned = [s[:, :, -min_len:] for s in skips]

            # Aggregate skips
            skip_cat = torch.cat(skips_aligned, dim=1)
            aggregated = self.skip_aggregate(skip_cat)

            # Channel attention
            aggregated = self.channel_attn(aggregated)

            # Dual pooling
            avg = self.avg_pool(aggregated).squeeze(-1)
            mx = self.max_pool(aggregated).squeeze(-1)
            pooled = torch.cat([avg, mx], dim=-1)

            # Project to latent
            latent = self.latent_proj(pooled)

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
