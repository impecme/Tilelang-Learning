"""GEMM references and TileLang learning kernels."""

from functools import lru_cache

import torch


def _dtype_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported dtype for matmul: {dtype}")
    return mapping[dtype]


def _load_tilelang():
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - exercised through optional smoke tests
        raise RuntimeError("tilelang is not installed or failed to import") from exc
    return tilelang, T


def _tl_dtype(T, dtype: str):
    mapping = {
        "float16": T.float16,
        "bfloat16": T.bfloat16,
        "float32": T.float32,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TileLang dtype: {dtype}")
    return mapping[dtype]


def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    return a @ b


@lru_cache(maxsize=32)
def _compile_gemm(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    num_stages: int,
    dtype: str,
    accum_dtype: str,
):
    tilelang, T = _load_tilelang()
    dtype_obj = _tl_dtype(T, dtype)
    accum_dtype_obj = _tl_dtype(T, accum_dtype)

    @tilelang.jit(out_idx=[-1])
    def _kernel_factory(
        M: int,
        N: int,
        K: int,
        block_M: int = 128,
        block_N: int = 128,
        block_K: int = 32,
        threads: int = 128,
        num_stages: int = 3,
        dtype_name=dtype_obj,
        accum_dtype_name=accum_dtype_obj,
    ):
        @T.prim_func
        def _kernel(
            A: T.Tensor((M, K), dtype_name),
            B: T.Tensor((K, N), dtype_name),
            C: T.Tensor((M, N), dtype_name),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype_name)
                B_shared = T.alloc_shared((block_K, block_N), dtype_name)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype_name)

                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return _kernel

    return _kernel_factory(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        threads,
        num_stages,
        dtype,
        accum_dtype,
    )


def matmul_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    threads: int = 128,
    num_stages: int = 3,
    accum_dtype: str = "float32",
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("TileLang GEMM expects CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("TileLang GEMM expects contiguous tensors")

    m, k = a.shape
    _, n = b.shape
    kernel = _compile_gemm(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        threads,
        num_stages,
        _dtype_name(a.dtype),
        accum_dtype,
    )
    return kernel(a, b)
