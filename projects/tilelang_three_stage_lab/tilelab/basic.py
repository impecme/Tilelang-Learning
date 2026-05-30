"""Stage 01 basic kernels.

The TileLang versions are intentionally small and readable. They are correctness-first
teaching kernels, not performance targets.
"""

from functools import lru_cache

import torch

from .common import (
    dtype_name,
    load_tilelang,
    require_cuda_contiguous,
    require_same_shape,
    require_single_block_reduction,
)


def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    require_same_shape(a, b)
    return a + b


def copy_reference(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def axpy_reference(alpha: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    require_same_shape(x, y)
    return float(alpha) * x + y


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return x.float().sum(dim=-1)


def row_sum_parallel_reference(x: torch.Tensor) -> torch.Tensor:
    return row_sum_reference(x)


@lru_cache(maxsize=64)
def _compile_vector_add(numel: int, block: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(
            A: T.Tensor((N,), dtype_name),
            B: T.Tensor((N,), dtype_name),
            C: T.Tensor((N,), dtype_name),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return _kernel

    return _factory(numel, block, dtype)


@lru_cache(maxsize=64)
def _compile_copy(numel: int, block: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(A: T.Tensor((N,), dtype_name), C: T.Tensor((N,), dtype_name)):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        C[idx] = A[idx]

        return _kernel

    return _factory(numel, block, dtype)


@lru_cache(maxsize=64)
def _compile_axpy(numel: int, block: int, dtype: str, alpha: float):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(
        N: int,
        block_size: int = 256,
        dtype_name: str = "float32",
        alpha_value: float = 1.0,
    ):
        @T.prim_func
        def _kernel(
            X: T.Tensor((N,), dtype_name),
            Y: T.Tensor((N,), dtype_name),
            C: T.Tensor((N,), dtype_name),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        C[idx] = X[idx] * alpha_value + Y[idx]

        return _kernel

    return _factory(numel, block, dtype, float(alpha))


@lru_cache(maxsize=64)
def _compile_row_sum(rows: int, cols: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((M, N), dtype_name), Y: T.Tensor((M,), "float32")):
            with T.Kernel(M, threads=1) as row:
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for col in T.serial(N):
                    acc[0] += X[row, col]
                Y[row] = acc[0]

        return _kernel

    return _factory(rows, cols, dtype)


@lru_cache(maxsize=64)
def _compile_row_sum_parallel(rows: int, cols: int, block_n: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, block: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((M, N), dtype_name), Y: T.Tensor((M,), "float32")):
            with T.Kernel(M, threads=block) as row:
                values = T.alloc_fragment((block,), "float32")
                total = T.alloc_fragment((1,), "float32")

                for col in T.Parallel(block):
                    if col < N:
                        values[col] = X[row, col]
                    else:
                        values[col] = 0.0

                T.reduce_sum(values, total, dim=0, clear=True)
                Y[row] = total[0]

        return _kernel

    return _factory(rows, cols, block_n, dtype)


def vector_add_tilelang(a: torch.Tensor, b: torch.Tensor, block: int = 256) -> torch.Tensor:
    require_same_shape(a, b)
    require_cuda_contiguous(a, b)
    out = torch.empty_like(a)
    kernel = _compile_vector_add(a.numel(), block, dtype_name(a.dtype))
    kernel(a.reshape(-1), b.reshape(-1), out.reshape(-1))
    return out


def copy_tilelang(x: torch.Tensor, block: int = 256) -> torch.Tensor:
    require_cuda_contiguous(x)
    out = torch.empty_like(x)
    kernel = _compile_copy(x.numel(), block, dtype_name(x.dtype))
    kernel(x.reshape(-1), out.reshape(-1))
    return out


def axpy_tilelang(alpha: float, x: torch.Tensor, y: torch.Tensor, block: int = 256) -> torch.Tensor:
    require_same_shape(x, y)
    require_cuda_contiguous(x, y)
    out = torch.empty_like(x)
    kernel = _compile_axpy(x.numel(), block, dtype_name(x.dtype), float(alpha))
    kernel(x.reshape(-1), y.reshape(-1), out.reshape(-1))
    return out


def row_sum_tilelang(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    require_cuda_contiguous(x)
    out = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    kernel = _compile_row_sum(x.shape[0], x.shape[1], dtype_name(x.dtype))
    kernel(x, out)
    return out


def row_sum_parallel_tilelang(x: torch.Tensor, block_n: int = 256) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    require_single_block_reduction("row_sum_parallel_tilelang", x.shape[1], block_n)
    require_cuda_contiguous(x)
    out = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    kernel = _compile_row_sum_parallel(x.shape[0], x.shape[1], block_n, dtype_name(x.dtype))
    kernel(x, out)
    return out
