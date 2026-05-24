"""PyTorch reference implementations used to validate TileLang kernels."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _check_attention_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B, H, S, D)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v must have the same number of heads")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("attention references in this project expect CUDA tensors")


def _attention_scale(q: torch.Tensor, sm_scale: float | None) -> float:
    return float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])


def naive_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Materialized attention reference.

    This is intentionally simple and memory hungry. It is the correctness oracle for
    the learning project, not the performance target.
    """

    _check_attention_inputs(q, k, v)
    scale = _attention_scale(q, sm_scale)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if causal:
        q_len, k_len = q.shape[-2], k.shape[-2]
        row_idx = torch.arange(q_len, device=q.device).view(q_len, 1)
        col_idx = torch.arange(k_len, device=q.device).view(1, k_len)
        scores = scores.masked_fill(col_idx > row_idx, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float())
    return out.to(dtype=q.dtype)


def sdpa_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """PyTorch scaled dot product attention baseline."""

    _check_attention_inputs(q, k, v)
    if sm_scale is None:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=causal,
        scale=float(sm_scale),
    )


def online_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_n: int = 64,
) -> torch.Tensor:
    """Online-softmax attention reference shaped like FlashAttention.

    The implementation stays in PyTorch so it is easy to inspect and test. The state
    update mirrors the algorithm the TileLang capstone kernel will implement:
    running max `m`, running denominator `l`, and running accumulator `acc`.
    """

    _check_attention_inputs(q, k, v)
    if block_n <= 0:
        raise ValueError("block_n must be positive")

    scale = _attention_scale(q, sm_scale)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    batch, heads, q_len, _ = q.shape
    k_len = k.shape[-2]
    value_dim = v.shape[-1]

    m = torch.full((batch, heads, q_len, 1), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((batch, heads, q_len, 1), device=q.device, dtype=torch.float32)
    acc = torch.zeros((batch, heads, q_len, value_dim), device=q.device, dtype=torch.float32)

    row_idx = torch.arange(q_len, device=q.device).view(1, 1, q_len, 1)

    for start in range(0, k_len, block_n):
        end = min(start + block_n, k_len)
        k_block = k_f[..., start:end, :]
        v_block = v_f[..., start:end, :]
        scores = torch.matmul(q_f, k_block.transpose(-2, -1)) * scale

        if causal:
            col_idx = torch.arange(start, end, device=q.device).view(1, 1, 1, end - start)
            scores = scores.masked_fill(col_idx > row_idx, float("-inf"))

        block_m = torch.max(scores, dim=-1, keepdim=True).values
        new_m = torch.maximum(m, block_m)
        old_scale = torch.exp(m - new_m)
        p = torch.exp(scores - new_m)

        l = l * old_scale + p.sum(dim=-1, keepdim=True)
        acc = acc * old_scale + torch.matmul(p, v_block)
        m = new_m

    tiny = torch.finfo(acc.dtype).tiny
    return (acc / l.clamp_min(tiny)).to(dtype=q.dtype)

