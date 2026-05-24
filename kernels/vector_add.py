"""Week 1 TileLang vector-add exercise."""

from functools import lru_cache

import torch


def _dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported dtype for vector add: {dtype}")
    return mapping[dtype]


def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    return a + b


def _load_tilelang():
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - exercised through optional smoke tests
        raise RuntimeError("tilelang is not installed or failed to import") from exc
    return tilelang, T


@lru_cache(maxsize=32)
def _compile_vector_add(numel: int, block: int, dtype: str):
    tilelang, T = _load_tilelang()

    @tilelang.jit
    def _kernel_factory(N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(
            A: T.Tensor((N,), dtype_name),
            B: T.Tensor((N,), dtype_name),
            C: T.Tensor((N,), dtype_name),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    C[idx] = A[idx] + B[idx]

        return _kernel

    return _kernel_factory(numel, block, dtype)


def vector_add_tilelang(a: torch.Tensor, b: torch.Tensor, block: int = 256) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("TileLang vector add expects CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("TileLang vector add expects contiguous tensors")

    out = torch.empty_like(a)
    kernel = _compile_vector_add(a.numel(), block, _dtype_name(a.dtype))
    kernel(a, b, out)
    return out
