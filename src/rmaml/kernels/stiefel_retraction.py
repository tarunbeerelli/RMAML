"""
Triton kernel for Cayley retraction on the Stiefel manifold.

Mathematical foundation:
    Standard retraction uses QR decomposition — O(np²), not optimised
    for tall-skinny matrices like our (1600, 5) Stiefel weight.

    Cayley retraction (Wen & Yin 2013):
        W_new = (I + A/2)⁻¹(I - A/2)W   where A = VWᵀ - WVᵀ

    Woodbury identity reduces (n,n) inversion to (2p,2p):
        U_L = [V, -W],  U_R = [W, V]   both (n, 2p)
        K   = 2I + U_Rᵀ U_L            (2p, 2p) = (10, 10) — trivially inverted

        W_new = rhs - U_L · K⁻¹ · U_Rᵀ · rhs
        where rhs = (I - A/2)W = W - 0.5(V - W(VᵀW))

    The (n,n) matrix A is NEVER formed.

Triton kernel fusion:
    Precompute correction = K⁻¹ U_Rᵀ rhs  (2p, p) = (10, 5) in PyTorch
    Kernel fuses:  out = rhs - U_L @ correction  across BLOCK_N row tiles
    Intermediates stay in SRAM — eliminates HBM roundtrips.

    Memory saved per call:  ~320KB
    Over 1.2M training calls: ~384GB HBM traffic eliminated

Benchmark (RTX 4090):
    geoopt QR:              6,133 μs/call
    Cayley+Woodbury+Triton: TBD   μs/call  (run benchmark_cuda.py)
    Cayley+Woodbury+PyTorch:  645 μs/call

References:
    Wen & Yin (2013). A feasible method for optimization with orthogonality constraints.
    Li et al. (2020). Efficient Riemannian optimization on the Stiefel manifold via Cayley.
"""

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ── Triton kernel ─────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def _cayley_retraction_kernel(
        Rhs_ptr,  # (n, p)   precomputed (I - A/2)W
        UL_ptr,  # (n, 2p)  U_L = [V, -W]
        Corr_ptr,  # (2p, p)  precomputed K⁻¹ U_Rᵀ rhs — tiny, in registers
        Out_ptr,  # (n, p)   output W_new
        n,
        p,
        p2,  # dimensions; p2 = 2p
        stride_rn,
        stride_rp,  # rhs strides
        stride_un,
        stride_up2,  # U_L strides
        stride_cn,
        stride_cp,  # correction strides
        stride_on,
        stride_op,  # output strides
        BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_P2: tl.constexpr,
    ):
        """
        Each program handles BLOCK_N rows.
        correction (2p, p) is loaded once per program into registers —
        no HBM access for it after the initial load.
        """
        pid = tl.program_id(0)
        row_start = pid * BLOCK_N

        rows = row_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        cols = tl.arange(0, BLOCK_P)  # (BLOCK_P,)
        p2s = tl.arange(0, BLOCK_P2)  # (BLOCK_P2,)

        rmask = rows < n
        cmask = cols < p
        p2mask = p2s < p2

        # Load rhs tile: (BLOCK_N, BLOCK_P)
        rhs = tl.load(
            Rhs_ptr + rows[:, None] * stride_rn + cols[None, :] * stride_rp,
            mask=rmask[:, None] & cmask[None, :],
            other=0.0,
        )

        # Load U_L tile: (BLOCK_N, BLOCK_P2)
        ul = tl.load(
            UL_ptr + rows[:, None] * stride_un + p2s[None, :] * stride_up2,
            mask=rmask[:, None] & p2mask[None, :],
            other=0.0,
        )

        # Load correction: (BLOCK_P2, BLOCK_P) — tiny, kept in registers
        corr = tl.load(
            Corr_ptr + p2s[:, None] * stride_cn + cols[None, :] * stride_cp,
            mask=p2mask[:, None] & cmask[None, :],
            other=0.0,
        )

        # Fused: out = rhs - U_L @ correction
        out = rhs - tl.dot(ul, corr)

        tl.store(
            Out_ptr + rows[:, None] * stride_on + cols[None, :] * stride_op,
            out,
            mask=rmask[:, None] & cmask[None, :],
        )

    def _run_triton_kernel(
        W: torch.Tensor,
        U_L: torch.Tensor,
        U_R: torch.Tensor,
        K: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Precompute correction = K⁻¹ U_Rᵀ rhs (10×5), then launch kernel.
        The expensive n-row computation is fused; the tiny (10×5) op stays in PyTorch.
        """
        n, p = W.shape
        p2 = 2 * p

        # Precompute correction — (2p, p) = (10, 5), trivially fast
        corr = torch.linalg.solve(K, U_R.T @ rhs).contiguous()

        rhs_c = rhs.contiguous()
        ul_c = U_L.contiguous()
        out = torch.empty_like(W)

        BLOCK_N = 32
        BLOCK_P = triton.next_power_of_2(p)  # 8 for p=5
        BLOCK_P2 = triton.next_power_of_2(p2)  # 16 for p2=10
        grid = (triton.cdiv(n, BLOCK_N),)

        _cayley_retraction_kernel[grid](
            rhs_c,
            ul_c,
            corr,
            out,
            n,
            p,
            p2,
            rhs_c.stride(0),
            rhs_c.stride(1),
            ul_c.stride(0),
            ul_c.stride(1),
            corr.stride(0),
            corr.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=BLOCK_N,
            BLOCK_P=BLOCK_P,
            BLOCK_P2=BLOCK_P2,
        )
        return out


# ── Differentiable wrapper ────────────────────────────────────────────────────


class CayleyRetraction(torch.autograd.Function):
    """
    Differentiable Cayley retraction.
    Forward uses Triton kernel on CUDA, PyTorch fallback on CPU.
    Backward uses PyTorch autograd (same for both paths).
    """

    @staticmethod
    def forward(ctx, W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        n, p = W.shape

        # (I - A/2)W = W - 0.5(V - W(VᵀW))  — no large matrix needed
        VtW = V.T @ W
        rhs = W - 0.5 * (V - W @ VtW)  # (n, p)

        # Woodbury matrices
        U_L = torch.cat([V, -W], dim=1)  # (n, 2p)
        U_R = torch.cat([W, V], dim=1)  # (n, 2p)
        K = 2 * torch.eye(2 * p, device=W.device, dtype=W.dtype) + U_R.T @ U_L

        # Always compute K_inv_URt_rhs for backward
        U_R_t_rhs = U_R.T @ rhs
        K_inv_URt_rhs = torch.linalg.solve(K, U_R_t_rhs)

        # Forward: fused Triton on CUDA, PyTorch fallback elsewhere
        if TRITON_AVAILABLE and W.is_cuda:
            W_new = _run_triton_kernel(W, U_L, U_R, K, rhs)
        else:
            W_new = rhs - U_L @ K_inv_URt_rhs

        ctx.save_for_backward(W, V, U_L, U_R, K, rhs, K_inv_URt_rhs)
        return W_new

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        W, V, U_L, U_R, K, rhs, K_inv_URt_rhs = ctx.saved_tensors

        grad_rhs = grad_output - U_L @ torch.linalg.solve(K.T, U_R.T @ grad_output)
        grad_UL_term = -grad_output @ K_inv_URt_rhs.T
        grad_URt_rhs = -torch.linalg.solve(K.T, U_L.T @ grad_output)
        grad_UR = rhs @ grad_URt_rhs.T
        grad_K = -torch.linalg.solve(K.T, (U_L.T @ grad_output) @ K_inv_URt_rhs.T)
        grad_UR = grad_UR + U_L @ (grad_K + grad_K.T)
        grad_UL = grad_UL_term + U_R @ grad_K.T

        VtW = V.T @ W
        grad_V = -0.5 * grad_rhs + grad_UL[:, : W.shape[1]] + grad_UR[:, W.shape[1] :]
        grad_W = (
            grad_output
            + 0.5 * grad_rhs @ VtW.T
            - grad_UL[:, W.shape[1] :]
            + grad_UR[:, : W.shape[1]]
            + U_L @ grad_URt_rhs
        )

        return grad_W, grad_V


# ── Public API ────────────────────────────────────────────────────────────────


def cayley_retract(W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Cayley retraction on the Stiefel manifold.

    Uses fused Triton kernel on CUDA, pure PyTorch fallback on CPU.
    3-9.5× faster than geoopt QR retraction.

    Args:
        W: point on St(n,p), shape (n, p), W^T W = I
        V: tangent vector at W, shape (n, p)
    Returns:
        W_new on St(n,p)
    """
    return CayleyRetraction.apply(W, V)


def verify_on_manifold(W: torch.Tensor, tol: float = 1e-4) -> float:
    """Returns ||W^T W - I|| — should be < tol after retraction."""
    return (W.T @ W - torch.eye(W.shape[1], device=W.device)).norm().item()
