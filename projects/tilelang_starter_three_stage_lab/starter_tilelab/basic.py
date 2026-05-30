"""Stage 01: the first useful TileLang kernel."""

from functools import lru_cache

import torch

from .common import dtype_name, load_tilelang, require_cuda_contiguous, require_same_shape


def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    require_same_shape(a, b)
    return a + b


@lru_cache(maxsize=32)
def _compile_vector_add(numel: int, block_size: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(N: int, block: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(
            A: T.Tensor((N,), dtype_name),
            B: T.Tensor((N,), dtype_name),
            C: T.Tensor((N,), dtype_name),
        ):
            with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
                for i in T.Parallel(block):
                    idx = bx * block + i
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return _kernel

    return _factory(numel, block_size, dtype)


def vector_add_tilelang(a: torch.Tensor, b: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    require_same_shape(a, b)
    require_cuda_contiguous(a, b)
    out = torch.empty_like(a)
    kernel = _compile_vector_add(a.numel(), block_size, dtype_name(a.dtype))
    kernel(a.reshape(-1), b.reshape(-1), out.reshape(-1))
    return out
