"""Triton fused matmul + reduce. GPU-only: import-safe on CPU, launch raises/skips."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # CPU laptop / CI
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _mm_reduce_kernel(a, b, out, M, N, K, stride_am, stride_bk, stride_om,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Fused matmul with a row-wise sum epilogue written to out[:, 0]:
        # the reduce-scatter epilogue shape used after a tensor-parallel Linear.
        pid = tl.program_id(0)
        rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            av = tl.load(a + (rm[:, None] * stride_am + (k + rk)[None, :]),
                         mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0.0)
            bv = tl.load(b + ((k + rk)[:, None] * stride_bk + rn[None, :]),
                         mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(av, bv)
        tl.store(out + rm[:, None] * stride_om + rn[None, :],
                 acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
        row_sum = tl.sum(acc, axis=1)
        tl.store(out + rm * stride_om + N, row_sum, mask=rm < M)  # reduced column


def fused_matmul_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """out = [a@b | rowsum(a@b)] — matmul + epilogue reduce in one kernel launch."""
    if not (HAS_TRITON and a.is_cuda):
        raise RuntimeError("fused_matmul_reduce needs CUDA + Triton; skipped on CPU")
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n + 1), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(m, 32),)
    _mm_reduce_kernel[grid](a, b, out, m, n, k, a.stride(0), b.stride(0), out.stride(0),
                            BLOCK_M=32, BLOCK_N=min(32, triton.next_power_of_2(n)), BLOCK_K=32)
    return out
