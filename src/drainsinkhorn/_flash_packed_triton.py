"""Candidate-batched Triton extension for FlashSinkhorn's fused LSE update.

The mathematical kernel follows the shifted-potential FlashSinkhorn v0.3.3
formulation (MIT licensed, https://github.com/ot-triton-lab/flash-sinkhorn).
Unlike the upstream two-dimensional API, this extension adds a leading
candidate program axis.  It imports the upstream tiled-dot and online-LSE
helpers so the score arithmetic stays aligned with the installed backend.

This module is imported lazily by :mod:`drainsinkhorn.flash_packed`; importing
the base package therefore does not require Triton or FlashSinkhorn.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flash_sinkhorn.kernels._triton_helpers import (
    _final_lse,
    _online_softmax_rescale,
    _tiled_dot,
)


@triton.jit
def _flashsinkhorn_lse_packed_kernel(
    x_ptr,
    y_ptr,
    g_hat_ptr,
    log_w_ptr,
    out_ptr,
    n,
    m,
    stride_xb,
    stride_x0,
    stride_x1,
    stride_yb,
    stride_y0,
    stride_y1,
    stride_gb,
    stride_g0,
    stride_wb,
    stride_w0,
    stride_ob,
    stride_o0,
    coord_scale,
    eps,
    damping,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_EXP2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one fused LSE update for every active candidate."""

    candidate = tl.program_id(0)
    pid_m = tl.program_id(1)
    x_base = x_ptr + candidate * stride_xb
    y_base = y_ptr + candidate * stride_yb
    g_base = g_hat_ptr + candidate * stride_gb
    w_base = log_w_ptr + candidate * stride_wb
    out_base = out_ptr + candidate * stride_ob

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n
    running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)

    log2e = 1.4426950408889634
    inv_eps = 1.0 / eps
    score_scale = coord_scale * inv_eps
    score_scale_log2 = score_scale * log2e

    for start_n in range(0, m, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < m
        g_hat = tl.load(
            g_base + offs_n * stride_g0,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        log_w = tl.load(
            w_base + offs_n * stride_w0,
            mask=mask_n,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        dot = _tiled_dot(
            x_base,
            y_base,
            offs_m,
            offs_n,
            stride_x0,
            stride_x1,
            stride_y0,
            stride_y1,
            D,
            mask_m,
            mask_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            ALLOW_TF32,
        )
        if USE_EXP2:
            bias = g_hat * (inv_eps * log2e) + log_w * log2e
            values = tl.fma(dot, score_scale_log2, bias[None, :])
        else:
            bias = g_hat * inv_eps + log_w
            values = dot * score_scale + bias[None, :]
        values = tl.where(mask_n[None, :], values, -float("inf"))
        new_max, rescale, weights = _online_softmax_rescale(
            values, running_max, USE_EXP2
        )
        running_sum = running_sum * rescale + tl.sum(weights, axis=1)
        running_max = new_max

    lse = _final_lse(running_max, running_sum, USE_EXP2)
    output = -eps * damping * lse
    tl.store(out_base + offs_m * stride_o0, output, mask=mask_m)


def _batch_layout(
    value: torch.Tensor, width: int, name: str
) -> tuple[int, int, int, int, int]:
    """Return ``(n, d, batch_stride, row_stride, dim_stride)``."""

    if value.ndim == 2:
        n, dimension = value.shape
        return int(n), int(dimension), 0, int(value.stride(0)), int(value.stride(1))
    if value.ndim == 3 and value.shape[0] == width:
        _width, n, dimension = value.shape
        return (
            int(n),
            int(dimension),
            int(value.stride(0)),
            int(value.stride(1)),
            int(value.stride(2)),
        )
    raise ValueError(f"{name} must have shape [n,d] or [width,n,d]")


def flashsinkhorn_lse_packed(
    x: torch.Tensor,
    y: torch.Tensor,
    g_hat: torch.Tensor,
    log_w: torch.Tensor,
    epsilon: float,
    *,
    cost_scale: float,
    allow_tf32: bool,
    use_exp2: bool,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the candidate-batched shifted-potential LSE update.

    ``out`` is an optional preallocated candidate-major output buffer.  It is
    deliberately strict: accepting a wrong layout or dtype here would make
    the ping-pong path silently fall back to an implicit allocation/copy.
    """

    if g_hat.ndim != 2 or log_w.shape != g_hat.shape:
        raise ValueError("g_hat and log_w must have shape [width,m]")
    width, m = (int(g_hat.shape[0]), int(g_hat.shape[1]))
    if width < 1 or m < 1:
        raise ValueError("packed width and support size must be positive")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("packed FlashSinkhorn requires CUDA tensors")
    if any(value.device != x.device for value in (y, g_hat, log_w)):
        raise ValueError("packed FlashSinkhorn tensors must share one CUDA device")

    x_work = x.float().contiguous()
    y_work = y.float().contiguous()
    g_work = g_hat.float().contiguous()
    w_work = log_w.float().contiguous()
    n, dimension, stride_xb, stride_x0, stride_x1 = _batch_layout(
        x_work, width, "x"
    )
    y_m, y_dimension, stride_yb, stride_y0, stride_y1 = _batch_layout(
        y_work, width, "y"
    )
    if y_m != m or y_dimension != dimension:
        raise ValueError("point-cloud and shifted-potential dimensions disagree")
    expected_shape = (width, n)
    if out is None:
        output = torch.empty(expected_shape, device=x.device, dtype=torch.float32)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor or None")
        if not out.is_cuda or out.device != x.device:
            raise ValueError("out must be a CUDA tensor on the input device")
        if out.dtype != torch.float32:
            raise TypeError("out must have dtype torch.float32")
        if out.ndim != 2 or tuple(out.shape) != expected_shape:
            raise ValueError(f"out must have shape {expected_shape}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous in candidate-major layout")
        input_ptrs = (x, y, g_hat, log_w)
        if any(out.data_ptr() == value.data_ptr() for value in input_ptrs):
            raise ValueError("out must not alias an LSE input")
        output = out
    grid = (width, triton.cdiv(n, block_m))
    _flashsinkhorn_lse_packed_kernel[grid](
        x_work,
        y_work,
        g_work,
        w_work,
        output,
        n,
        m,
        stride_xb,
        stride_x0,
        stride_x1,
        stride_yb,
        stride_y0,
        stride_y1,
        g_work.stride(0),
        g_work.stride(1),
        w_work.stride(0),
        w_work.stride(1),
        output.stride(0),
        output.stride(1),
        float(2.0 * cost_scale),
        float(epsilon),
        1.0,
        D=dimension,
        ALLOW_TF32=allow_tf32,
        USE_EXP2=use_exp2,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


__all__ = ["flashsinkhorn_lse_packed"]
