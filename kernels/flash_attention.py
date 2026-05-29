"""Capstone-facing FlashAttention forward interface."""

from __future__ import annotations

import torch

from .reference import online_attention_forward


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """FlashAttention forward API for the learning project.

    The interface is fixed now. The current implementation is a PyTorch online
    softmax reference so tests and benchmarks can be written immediately. In Stage 07
    this body should be replaced by a TileLang kernel while preserving the signature.
    """

    return online_attention_forward(q, k, v, causal=causal, sm_scale=sm_scale)
