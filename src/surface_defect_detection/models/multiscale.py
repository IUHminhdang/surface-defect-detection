from __future__ import annotations

import torch
from torch import nn


class MultiScaleFusion(nn.Module):
    """Lightweight parallel 1x1, 3x3, and dilated 3x3 feature fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = max(channels // 4, 8)
        self.branches = nn.ModuleList(
            [
                self._branch(channels, branch_channels, 1, 1),
                self._branch(channels, branch_channels, 3, 1),
                self._branch(channels, branch_channels, 3, 2),
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _branch(channels: int, output_channels: int, kernel: int, dilation: int) -> nn.Sequential:
        padding = dilation if kernel == 3 else 0
        return nn.Sequential(
            nn.Conv2d(channels, output_channels, kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
