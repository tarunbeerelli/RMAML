"""
Stiefel linear layer — FC layer whose weight matrix lies on the
Stiefel manifold St(n, p): W ∈ R^(n x p) with W^T W = I.

Key constraint: n >= p  (more rows than columns).
For 5-way classification with 1600-dim features: St(1600, 5).
"""

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StiefelLinear(nn.Module):
    """
    Linear layer with weight on the Stiefel manifold.

    Args:
        in_features:  input dimension  (n in St(n,p)) — must be >= out_features
        out_features: output dimension (p in St(n,p)) — number of classes
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert in_features >= out_features, (
            f"Stiefel requires in_features >= out_features, "
            f"got {in_features} < {out_features}"
        )
        manifold = geoopt.Stiefel()

        # Initialise via QR decomposition of random Gaussian matrix
        # QR gives us an orthogonal matrix — valid starting point on St(n,p)
        W0 = torch.randn(in_features, out_features)
        Q, _ = torch.linalg.qr(W0)  # Q shape: (in_features, out_features)

        self.weight = geoopt.ManifoldParameter(Q, manifold=manifold)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        # weight is (in_features, out_features), F.linear expects (out, in)
        return F.linear(x, self.weight.T, self.bias)

    def orthogonality_error(self) -> float:
        """
        Diagnostic: measures how far W^T W is from identity.
        Should stay near 0 during training. If it grows > 1e-4,
        something is wrong with the Riemannian updates.
        """
        W = self.weight
        return (W.T @ W - torch.eye(W.size(1), device=W.device)).norm().item()
