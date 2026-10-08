from __future__ import annotations

import torch
from torch import nn


class P2HighResolution(nn.Module):
    """Project and preserve a stride-4 feature for the P2 detection pathway."""

    def __init__(self, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
