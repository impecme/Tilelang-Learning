"""Stage 02: the first matrix operator."""

from functools import lru_cache

import torch

from .common import dtype_name, load_tilelang, require_cuda_contiguous


def _tl_dtype(T, dtype: str):
    mapping = {
        "float16": T.float16,
        "bfloat16": T.bfloat16,
        "float32": T.float32,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TileLang dtype: {dtype}")
    return mapping[dtype]


def gemm_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    return a @ b


def _require_tile_aligned(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    errors = []
    if m % block_m != 0:
        errors.append(f"M={m} is not divisible by block_m={block_m}")
    if n % block_n != 0:
        errors.append(f"N={n} is not divisible by block_n={block_n}")
    if k % block_k != 0:
        errors.append(f"K={k} is not divisible by block_k={block_k}")
    if errors:
        raise ValueError(
            "starter gemm_tilelang requires tile-aligned shapes; "
            + "; ".join(errors)
        )


@lru_cache(maxsize=32)
def _compile_gemm(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    threads: int,
    dtype: str,
    accum_dtype: str,
):
    tilelang, T = load_tilelang()
    dtype_obj = _tl_dtype(T, dtype)
    accum_dtype_obj = _tl_dtype(T, accum_dtype)

    @tilelang.jit(out_idx=[-1])
    def _factory(
        M: int,
        N: int,
        K: int,
        block_M: int = 16,
        block_N: int = 16,
        block_K: int = 32,
        threads: int = 32,
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
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return _kernel

    return _factory(m, n, k, block_m, block_n, block_k, threads, dtype, accum_dtype)


@lru_cache(maxsize=32)
def _compile_gemm_serial(m: int, n: int, k: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, K: int, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(
            A: T.Tensor((M, K), dtype_name),
            B: T.Tensor((K, N), dtype_name),
            C: T.Tensor((M, N), dtype_name),
        ):
            with T.Kernel(N, M, threads=1) as (col, row):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = 0.0
                for kk in T.serial(K):
                    acc[0] += A[row, kk] * B[kk, col]
                C[row, col] = acc[0]

        return _kernel

    return _factory(m, n, k, dtype)


def gemm_tilelang(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 32,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    _require_tile_aligned(a.shape[0], b.shape[1], a.shape[1], block_m, block_n, block_k)
    require_cuda_contiguous(a, b)
    if a.shape[1] != block_k:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        kernel = _compile_gemm_serial(a.shape[0], b.shape[1], a.shape[1], dtype_name(a.dtype))
        kernel(a, b, out)
        return out
    kernel = _compile_gemm(
        a.shape[0],
        b.shape[1],
        a.shape[1],
        block_m,
        block_n,
        block_k,
        32,
        dtype_name(a.dtype),
        "float32",
    )
    return kernel(a, b)
