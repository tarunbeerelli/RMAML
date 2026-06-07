"""
Tests for backbone, Stiefel layer, and full RMAML model.
All CPU, no dataset needed.
"""

import pytest
import torch
from rmaml.models.backbone import Conv4
from rmaml.models.rmaml_model import RMAMLModel
from rmaml.models.stiefel_layer import StiefelLinear

# ── Conv4 backbone ────────────────────────────────────────────────


def test_conv4_output_shape():
    """84x84 input should produce 1600-dim features."""
    model = Conv4()
    x = torch.randn(4, 3, 84, 84)  # batch of 4
    out = model(x)
    assert out.shape == (4, 1600)


def test_conv4_out_dim():
    assert Conv4().out_dim == 1600


# ── StiefelLinear ─────────────────────────────────────────────────


def test_stiefel_init_orthogonal():
    """Weight should be orthogonal at initialisation."""
    layer = StiefelLinear(1600, 5)
    err = layer.orthogonality_error()
    assert err < 1e-5, f"Not orthogonal at init: {err:.2e}"


def test_stiefel_forward_shape():
    layer = StiefelLinear(1600, 5)
    x = torch.randn(8, 1600)
    assert layer(x).shape == (8, 5)


def test_stiefel_wrong_dims_raises():
    with pytest.raises(AssertionError):
        StiefelLinear(4, 10)  # out > in — invalid


def test_stiefel_grad_flows():
    layer = StiefelLinear(16, 5)
    x = torch.randn(4, 16)
    loss = layer(x).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None


# ── Full RMAMLModel ───────────────────────────────────────────────


def test_rmaml_forward_shape():
    model = RMAMLModel(n_way=5)
    x = torch.randn(4, 3, 84, 84)
    out = model(x)
    assert out.shape == (4, 5)


def test_rmaml_functional_forward():
    """functional_forward with cloned params should match forward."""
    model = RMAMLModel(n_way=5)
    x = torch.randn(2, 3, 84, 84)
    params = {n: p.clone() for n, p in model.named_parameters()}
    out = model.functional_forward(x, params)
    assert out.shape == (2, 5)


def test_rmaml_orthogonality_error():
    model = RMAMLModel(n_way=5)
    err = model.orthogonality_error()
    assert err < 1e-5
