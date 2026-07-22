"""
Benchmark Cayley retraction on CUDA.

Run on GPU machine:
    PYTHONPATH=src python benchmark_cuda.py

Compares three implementations:
    1. geoopt QR          — baseline
    2. Cayley PyTorch     — Woodbury math, no kernel fusion
    3. Cayley Triton      — Woodbury math + fused Triton kernel
"""

import time

import geoopt
import torch
from rmaml.kernels.stiefel_retraction import (
    TRITON_AVAILABLE,
    cayley_retract,
)

assert torch.cuda.is_available(), "CUDA required for this benchmark"
assert TRITON_AVAILABLE, "Triton required — install with: pip install triton"

DEVICE = torch.device("cuda")
N_WARMUP = 50
N_RUNS = 1000

stiefel = geoopt.Stiefel()


def benchmark(fn, *args, label=""):
    # Warmup
    for _ in range(N_WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_RUNS):
        fn(*args)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / N_RUNS

    print(f"  {label:<35} {elapsed_us:>8.1f} μs/call")
    return elapsed_us


def run_pytorch_only(W, V):
    """Cayley+Woodbury in pure PyTorch — no Triton kernel."""
    n, p = W.shape
    VtW = V.T @ W
    rhs = W - 0.5 * (V - W @ VtW)
    U_L = torch.cat([V, -W], dim=1)
    U_R = torch.cat([W, V], dim=1)
    K = 2 * torch.eye(2 * p, device=W.device, dtype=W.dtype) + U_R.T @ U_L
    K_inv_URt_rhs = torch.linalg.solve(K, U_R.T @ rhs)
    return rhs - U_L @ K_inv_URt_rhs


print("=" * 60)
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Warmup: {N_WARMUP} | Runs: {N_RUNS}")
print("=" * 60)

for n, p in [(1600, 5), (512, 5)]:
    print(f"\nShape ({n}, {p}):")
    W = stiefel.random(n, p).to(DEVICE)
    G = torch.randn(n, p, device=DEVICE)
    V = stiefel.egrad2rgrad(W, G)

    t_geoopt = benchmark(
        lambda: stiefel.retr(W, -0.01 * V), label="geoopt QR (baseline)"
    )
    t_pytorch = benchmark(
        lambda: run_pytorch_only(W, -0.01 * V), label="Cayley+Woodbury (PyTorch)"
    )
    t_triton = benchmark(
        lambda: cayley_retract(W, -0.01 * V), label="Cayley+Woodbury (Triton fused)"
    )

    print(f"  {'─'*50}")
    print(f"  PyTorch speedup vs geoopt: {t_geoopt/t_pytorch:.1f}x")
    print(f"  Triton  speedup vs geoopt: {t_geoopt/t_triton:.1f}x")
    print(f"  Triton  speedup vs PyTorch:{t_pytorch/t_triton:.1f}x")

print("\n" + "=" * 60)
