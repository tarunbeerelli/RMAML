"""
Full RMAML model: Conv4 backbone + Stiefel classification head.
The only Riemannian parameter is the final FC layer weight.
All conv/BN parameters remain Euclidean.
"""

import torch.nn as nn
import torch.nn.functional as F
from rmaml.models.backbone import Conv4
from rmaml.models.stiefel_layer import StiefelLinear
from torch import Tensor


class RMAMLModel(nn.Module):
    """
    Conv4 + Stiefel head for N-way few-shot classification.

    Args:
        n_way: number of classes per episode (default 5)
    """

    def __init__(self, n_way: int = 5):
        super().__init__()
        self.encoder = Conv4()
        self.classifier = StiefelLinear(
            in_features=self.encoder.out_dim,  # 1600
            out_features=n_way,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))

    def functional_forward(self, x: Tensor, params: dict) -> Tensor:
        """
        Forward with external param dict — used during inner loop
        so gradients flow through task-specific adapted parameters.
        """
        h = self.encoder.functional_forward(x, params)
        # weight is (in, out) on Stiefel — transpose for F.linear
        return F.linear(h, params["classifier.weight"].T, params["classifier.bias"])

    def orthogonality_error(self) -> float:
        """Convenience wrapper — log this during training."""
        return self.classifier.orthogonality_error()
