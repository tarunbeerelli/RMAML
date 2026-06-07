import geoopt
import torch


def test_torch_available():
    assert torch.__version__ is not None


def test_geoopt_stiefel_basic():
    manifold = geoopt.Stiefel()
    W = manifold.random(10, 4)
    G = torch.randn(10, 4)
    rgrad = manifold.egrad2rgrad(W, G)
    W_new = manifold.retr(W, -0.01 * rgrad)
    err = (W_new.T @ W_new - torch.eye(4)).norm().item()
    assert err < 1e-5, f"Orthogonality error too large: {err:.2e}"


def test_geoopt_manifold_parameter():
    manifold = geoopt.Stiefel()
    W = torch.randn(8, 3)
    Q, _ = torch.linalg.qr(W)
    param = geoopt.ManifoldParameter(Q, manifold=manifold)
    assert isinstance(param, torch.nn.Parameter)
    assert param.shape == (8, 3)
