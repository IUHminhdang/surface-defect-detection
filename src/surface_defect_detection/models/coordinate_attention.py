from __future__ import annotations

import torch
from torch import nn


class CoordinateAttention(nn.Module):
    """Coordinate Attention from pooled height and width context."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        self.attn_h = nn.Conv2d(hidden, channels, 1)
        self.attn_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        height, width = x.shape[-2:]
        pooled_h = self.pool_h(x)
        pooled_w = self.pool_w(x).transpose(2, 3)
        context = self.shared(torch.cat([pooled_h, pooled_w], dim=2))
        context_h, context_w = torch.split(context, [height, width], dim=2)
        attention_h = self.attn_h(context_h).sigmoid()
        attention_w = self.attn_w(context_w.transpose(2, 3)).sigmoid()
        return identity * attention_h * attention_w
