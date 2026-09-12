"""Lightweight 1D CNN for RF modulation classification.

Contract (agreed shared interface):
    * input  x : float tensor of shape [B, 2, 128]  (IQ signal, 2 channels)
    * output   : logits of shape [B, num_classes]   (num_classes == 11)

The model depends on ``x`` ONLY. It never sees SNR or task id -- that keeps the
backbone reusable across every baseline and continual-learning method.

Nothing here is claimed to be novel; the research contribution lives in the
continual-learning strategy, not in this backbone.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU (-> optional MaxPool)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = False,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # 'same' length for odd kernels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class RFClassifier(nn.Module):
    """A small, configurable 1D CNN.

    Architecture (default ``conv_channels=(64, 64, 128)``)::

        [B, 2, 128]
          -> ConvBlock(2   -> 64)                 # no pool
          -> ConvBlock(64  -> 64)  + MaxPool
          -> ConvBlock(64  -> 128) + MaxPool
          -> AdaptiveAvgPool1d(1)                 # global pooling -> [B, 128]
          -> Linear(128 -> fc_hidden) -> ReLU -> Dropout
          -> Linear(fc_hidden -> num_classes)     # logits

    All width/depth knobs are configurable so the same class covers the
    "make architecture configurable where practical" requirement while still
    working out-of-the-box as ``RFClassifier(num_classes=11)``.
    """

    def __init__(
        self,
        num_classes: int = 11,
        in_channels: int = 2,
        conv_channels: Sequence[int] = (64, 64, 128),
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        conv_channels = list(conv_channels)
        if len(conv_channels) < 1:
            raise ValueError("conv_channels must contain at least one entry")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        # Feature extractor: pool after every block except the first, matching
        # the recommended architecture for a 3-block stack.
        blocks = []
        prev = in_channels
        for i, ch in enumerate(conv_channels):
            blocks.append(
                ConvBlock(prev, ch, kernel_size=kernel_size, pool=(i > 0), pool_size=pool_size)
            )
            prev = ch
        self.features = nn.Sequential(*blocks)

        # Global pooling makes the head independent of the input length.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        head: list[nn.Module] = []
        if fc_hidden and fc_hidden > 0:
            head.append(nn.Linear(prev, fc_hidden))
            head.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                head.append(nn.Dropout(float(dropout)))
            head.append(nn.Linear(fc_hidden, self.num_classes))
        else:
            head.append(nn.Linear(prev, self.num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected input of shape [B, C, L], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.size(1)}"
            )
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)  # [B, C, 1] -> [B, C]
        return self.classifier(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def resolve_num_classes(config: dict | None = None, default: int = 11) -> int:
    """Find ``num_classes`` wherever the shared config puts it.

    The shared ``config.yaml`` may declare it top-level, under ``dataset:``
    (Person 1) or under ``evaluation:`` (Person 3). We read it rather than
    hard-coding 11, but keep 11 as the ultimate fallback.
    """
    config = config or {}
    for holder in (config, config.get("dataset") or {}, config.get("evaluation") or {}):
        val = holder.get("num_classes")
        if val is not None:
            return int(val)
    return int(default)


def build_model(config: dict | None = None) -> RFClassifier:
    """Factory that builds an :class:`RFClassifier` from a parsed config dict.

    Keeps experiment code free of magic numbers -- every knob comes from
    ``config.yaml``. Unknown/missing keys fall back to sensible defaults.
    """
    config = config or {}
    m = dict(config.get("model", {}))
    return RFClassifier(
        num_classes=resolve_num_classes(config),
        in_channels=int(m.get("in_channels", 2)),
        conv_channels=m.get("conv_channels", (64, 64, 128)),
        kernel_size=int(m.get("kernel_size", 3)),
        pool_size=int(m.get("pool_size", 2)),
        fc_hidden=int(m.get("fc_hidden", 128)),
        dropout=float(m.get("dropout", 0.3)),
    )
