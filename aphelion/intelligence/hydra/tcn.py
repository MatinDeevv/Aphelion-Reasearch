"""
APHELION HYDRA — Temporal Convolutional Network (Phase 7 v2)
Long-range temporal dependencies via dilated causal convolutions.
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
    hidden_size: int = 128
    num_channels: list[int] = None  # Channel sizes per layer
    kernel_size: int = 3
    dropout: float = 0.2
    n_classes: int = 3
    n_horizons: int = 3
    sequence_length: int = 512

    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [64, 128, 128, 256]


if HAS_TORCH:
    class CausalConv1d(nn.Module):
        """Causal convolution with left-padding to prevent future leakage."""

        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, dilation: int, dropout: float = 0.2):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, (self.padding, 0))
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x

    class TemporalBlock(nn.Module):
        """Residual block with two causal dilated convolutions."""

        def __init__(self, in_ch: int, out_ch: int,
                     kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation, dropout)
            self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation, dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x if self.downsample is None else self.downsample(x)
            out = self.conv1(x)
            out = self.conv2(out)
            return F.relu(out + residual)

    class HydraTCN(nn.Module):
        """Temporal Convolutional Network for HYDRA ensemble."""

        def __init__(self, config: Optional[TCNConfig] = None):
            super().__init__()
            self.config = config or TCNConfig()
            cfg = self.config

            self.input_proj = nn.Linear(cfg.input_size, cfg.num_channels[0])

            layers = []
            channels = cfg.num_channels
            for i in range(len(channels)):
                in_ch = channels[i - 1] if i > 0 else channels[0]
                out_ch = channels[i]
                dilation = 2 ** i
                layers.append(TemporalBlock(in_ch, out_ch, cfg.kernel_size, dilation, cfg.dropout))
            self.network = nn.Sequential(*layers)

            self.latent_proj = nn.Linear(channels[-1], cfg.hidden_size)

            self.aux_head = nn.Linear(cfg.hidden_size, cfg.n_classes * cfg.n_horizons)
            self._n_classes = cfg.n_classes
            self._n_horizons = cfg.n_horizons

        def forward(self, cont_inputs: torch.Tensor,
                    cat_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            # cont_inputs: (batch, seq, features)
            x = self.input_proj(cont_inputs)  # (batch, seq, channels[0])
            x = x.transpose(1, 2)  # (batch, channels, seq) for Conv1d
            x = self.network(x)
            x = x[:, :, -1]  # Take last time step: (batch, channels[-1])

            latent = self.latent_proj(x)  # (batch, hidden_size)

            aux = self.aux_head(latent)
            aux = aux.view(-1, self._n_horizons, self._n_classes)

            return {
                "latent": latent,
                "aux_logits": aux[:, 0, :],
                "aux_logits_all": aux,
            }

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
