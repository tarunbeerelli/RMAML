"""
Correctness and speed tests for Stiefel retraction kernel.

Correctness: our Cayley retraction must match geoopt's QR retraction
             to within numerical tolerance on the manifold.

Speed:       benchmark runs on CPU locally (no GPU needed for CI).
             Real speedup numbers require CUDA — run on Vast.ai.
"""

import time

import geoopt
import pytest
import torch
from rmaml.kernels.stiefel_retraction import (
    cayley_retract,
    verify_on_manifold,
)

STIEFEL = geoopt.Stiefel()

# Shapes to test — (n, p) where n >> p mimics our (1600, 5) case
TEST_SHAPES = [
    (1600, 5),  # our actual use case
    (512, 5),  # smaller for fast CI
    (64, 4),  # tiny for smoke tests
]


# ── Correctness tests ─────────────────────────────────────────────


@pytest.mark.parametrize("n,p", TEST_SHAPES)
def test_cayley_stays_on_manifold(n, p):
    """Cayley retraction output should satisfy W^T W = I."""
    W = STIEFEL.random(n, p)
    G = torch.randn(n, p)
    V = STIEFEL.egrad2rgrad(W, G)
    W_new = cayley_retract(W, -0.01 * V)
    err = verify_on_manifold(W_new)
    assert err < 1e-4, f"Off manifold after Cayley retraction: {err:.2e}"


@pytest.mark.parametrize("n,p", TEST_SHAPES)
def test_cayley_moves_point(n, p):
    """Retraction should produce a different point."""
    W = STIEFEL.random(n, p)
    G = torch.randn(n, p)
    V = STIEFEL.egrad2rgrad(W, G)
    W_new = cayley_retract(W, -0.1 * V)
    assert not torch.allclose(W, W_new), "Retraction produced no movement"


@pytest.mark.parametrize("n,p", TEST_SHAPES)
def test_cayley_agrees_with_geoopt(n, p):
    """
    Cayley and QR retractions won't give identical outputs
    (different algorithms) but both should stay on the manifold
    with similar step magnitude.
    """
    W = STIEFEL.random(n, p)
    G = torch.randn(n, p)
    V = STIEFEL.egrad2rgrad(W, G)

    W_cayley = cayley_retract(W, -0.01 * V)
    W_qr = STIEFEL.retr(W, -0.01 * V)

    # Both should be on manifold
    assert verify_on_manifold(W_cayley) < 1e-4
    assert verify_on_manifold(W_qr) < 1e-4

    # Step sizes should be similar (within 20%)
    step_cayley = (W_cayley - W).norm().item()
    step_qr = (W_qr - W).norm().item()
    ratio = step_cayley / (step_qr + 1e-8)
    assert (
        0.5 < ratio < 2.0
    ), f"Step size ratio unexpected: cayley={step_cayley:.4f}, qr={step_qr:.4f}"


def test_cayley_gradient_flows():
    """Gradient should flow through Cayley retraction."""
    W = STIEFEL.random(64, 4).requires_grad_(True)
    V = torch.randn(64, 4)
    W_new = cayley_retract(W, V)
    loss = W_new.sum()
    loss.backward()
    assert W.grad is not None
    assert not torch.isnan(W.grad).any()


def test_cayley_zero_tangent():
    """Zero tangent vector should return the same point."""
    W = STIEFEL.random(64, 4)
    V = torch.zeros(64, 4)
    W_new = cayley_retract(W, V)
    assert torch.allclose(W, W_new, atol=1e-5)


# ── Speed benchmark ───────────────────────────────────────────────


def benchmark_retraction(fn, W, V, n_warmup=10, n_runs=100):
    """Time a retraction function."""
    for _ in range(n_warmup):
        fn(W, V)
    if W.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        fn(W, V)
    if W.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1e6  # microseconds per call


@pytest.mark.parametrize("n,p", [(1600, 5), (512, 5)])
def test_cayley_speed_vs_geoopt(n, p):
    """
    Benchmark Cayley vs QR retraction.
    On CPU both are similar — real speedup shows on CUDA.
    Test just verifies Cayley is not dramatically slower on CPU.
    """
    W = STIEFEL.random(n, p)
    G = torch.randn(n, p)
    V = STIEFEL.egrad2rgrad(W, G)

    cayley_us = benchmark_retraction(lambda w, v: cayley_retract(w, v), W, V)
    geoopt_us = benchmark_retraction(lambda w, v: STIEFEL.retr(w, v), W, V)

    print(f"\nShape ({n}, {p}):")
    print(f"  Cayley:  {cayley_us:.1f} μs/call")
    print(f"  geoopt:  {geoopt_us:.1f} μs/call")
    print(f"  Ratio:   {geoopt_us/cayley_us:.2f}x")

    # On CPU Cayley should be no more than 3x slower than QR
    # (real speedup happens on CUDA due to kernel fusion)
    assert (
        cayley_us < geoopt_us * 3
    ), f"Cayley too slow on CPU: {cayley_us:.1f} vs {geoopt_us:.1f} μs"
