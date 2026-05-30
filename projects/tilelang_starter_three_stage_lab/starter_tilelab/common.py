"""Shared helpers for the lightweight starter project."""

from __future__ import annotations

import torch


SUPPORTED_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def dtype_name(dtype: torch.dtype) -> str:
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype: {dtype}")
    return SUPPORTED_DTYPES[dtype]


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype name: {name}")
    return mapping[name]


def require_same_shape(*tensors: torch.Tensor) -> None:
    if not tensors:
        return
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError("all tensors must have the same shape")


def require_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("TileLang starter kernels expect CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("TileLang starter kernels expect contiguous tensors")


def load_tilelang():
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - exercised by optional smoke tests
        raise RuntimeError("tilelang is not installed or failed to import") from exc
    return tilelang, T
