"""
Conv4 backbone for few-shot classification.
Input:  (B, 3, 84, 84)
Output: (B, 1600)  — 64 channels * 5 * 5 spatial after 4x maxpool
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """One conv block: Conv → BN → ReLU → MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Conv4(nn.Module):
    """
    4-layer convolutional backbone used in MAML and RMAML paper.
    84x84 input → 4 conv blocks → flatten to 1600-dim vector.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
        )
        self.out_dim = hidden_dim * 5 * 5  # 1600

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x).view(x.size(0), -1)

    def functional_forward(self, x: Tensor, params: dict) -> Tensor:
        """
        Forward pass with external params — needed for inner loop
        where we maintain a separate param dict per task.
        """
        h = x
        for i in range(4):
            h = F.conv2d(
                h,
                params[f"encoder.encoder.{i}.0.weight"],
                padding=1,
            )
            h = F.batch_norm(
                h,
                running_mean=None,
                running_var=None,
                weight=params[f"encoder.encoder.{i}.1.weight"],
                bias=params[f"encoder.encoder.{i}.1.bias"],
                training=True,
            )
            h = F.relu(h, inplace=False)
            h = F.max_pool2d(h, 2)
        return h.view(h.size(0), -1)
