"""Stage 02 advanced operators for the TileLang learning lab."""

from functools import lru_cache

import torch

from .common import (
    dtype_name,
    gelu_torch,
    load_tilelang,
    require_cuda_contiguous,
    require_single_block_reduction,
)


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


def row_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return torch.softmax(x.float(), dim=-1).to(dtype=x.dtype)


def row_softmax_parallel_reference(x: torch.Tensor) -> torch.Tensor:
    return row_softmax_reference(x)


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must have shape (N,)")
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    out = x.float() * torch.rsqrt(variance + eps) * weight.float()
    return out.to(dtype=x.dtype)


def rmsnorm_parallel_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return rmsnorm_reference(x, weight, eps)


def gelu_reference(x: torch.Tensor) -> torch.Tensor:
    return gelu_torch(x.float()).to(dtype=x.dtype)


def causal_mask_reference(scores: torch.Tensor, seq_len: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2")
    if scores.shape[1] != seq_len:
        raise ValueError("scores.shape[1] must equal seq_len")
    rows = scores.shape[0]
    row_pos = torch.arange(rows, device=scores.device).remainder(seq_len).view(rows, 1)
    col_pos = torch.arange(seq_len, device=scores.device).view(1, seq_len)
    return scores.masked_fill(col_pos > row_pos, -65504.0)


def scale_reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * float(scale)


def add_bias_reference(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or bias.ndim != 1:
        raise ValueError("x must be rank-2 and bias must be rank-1")
    if x.shape[1] != bias.shape[0]:
        raise ValueError("expected x=(M,N), bias=(N,)")
    return x + bias


def linear_bias_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("x and weight must be rank-2, bias must be rank-1")
    if x.shape[1] != weight.shape[0] or weight.shape[1] != bias.shape[0]:
        raise ValueError("expected x=(M,K), weight=(K,N), bias=(N,)")
    return x @ weight + bias


def scale_causal_mask_reference(scores: torch.Tensor, scale: float, seq_len: int) -> torch.Tensor:
    return causal_mask_reference(scale_reference(scores, scale), seq_len)


def linear_bias_gelu_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("x and weight must be rank-2, bias must be rank-1")
    if x.shape[1] != weight.shape[0] or weight.shape[1] != bias.shape[0]:
        raise ValueError("expected x=(M,K), weight=(K,N), bias=(N,)")
    return gelu_reference(x @ weight + bias)


def _require_gemm_tile_aligned(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    mismatches = []
    if m % block_m != 0:
        mismatches.append(f"M={m} is not divisible by block_m={block_m}")
    if n % block_n != 0:
        mismatches.append(f"N={n} is not divisible by block_n={block_n}")
    if k % block_k != 0:
        mismatches.append(f"K={k} is not divisible by block_k={block_k}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(
            "TileLang GEMM v1 requires tile-aligned shapes because this teaching "
            f"kernel does not implement tail-tile guards: {detail}"
        )


@lru_cache(maxsize=64)
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
    tilelang, T = load_tilelang()
    dtype_obj = _tl_dtype(T, dtype)
    accum_dtype_obj = _tl_dtype(T, accum_dtype)

    @tilelang.jit(out_idx=[-1])
    def _factory(
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

    return _factory(m, n, k, block_m, block_n, block_k, threads, num_stages, dtype, accum_dtype)


@lru_cache(maxsize=64)
def _compile_row_softmax(rows: int, cols: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((M, N), dtype_name), Y: T.Tensor((M, N), dtype_name)):
            with T.Kernel(M, threads=1) as row:
                max_val = T.alloc_fragment((1,), "float32")
                sum_val = T.alloc_fragment((1,), "float32")
                max_val[0] = -3.402823e38
                for col in T.serial(N):
                    max_val[0] = T.max(max_val[0], X[row, col])
                sum_val[0] = 0.0
                for col in T.serial(N):
                    sum_val[0] += T.exp(X[row, col] - max_val[0])
                for col in T.serial(N):
                    Y[row, col] = T.exp(X[row, col] - max_val[0]) / sum_val[0]

        return _kernel

    return _factory(rows, cols, dtype)


@lru_cache(maxsize=64)
def _compile_rmsnorm(rows: int, cols: int, dtype: str, eps: float):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, dtype_name: str = "float32", eps_value: float = 1e-5):
        @T.prim_func
        def _kernel(
            X: T.Tensor((M, N), dtype_name),
            W: T.Tensor((N,), dtype_name),
            Y: T.Tensor((M, N), dtype_name),
        ):
            with T.Kernel(M, threads=1) as row:
                ss = T.alloc_fragment((1,), "float32")
                scale = T.alloc_fragment((1,), "float32")
                ss[0] = 0.0
                for col in T.serial(N):
                    ss[0] += X[row, col] * X[row, col]
                scale[0] = T.rsqrt(ss[0] / N + eps_value)
                for col in T.serial(N):
                    Y[row, col] = X[row, col] * scale[0] * W[col]

        return _kernel

    return _factory(rows, cols, dtype, float(eps))


@lru_cache(maxsize=64)
def _compile_row_softmax_parallel(rows: int, cols: int, block_n: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, block: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((M, N), dtype_name), Y: T.Tensor((M, N), dtype_name)):
            with T.Kernel(M, threads=block) as row:
                values = T.alloc_fragment((block,), "float32")
                exps = T.alloc_fragment((block,), "float32")
                max_val = T.alloc_fragment((1,), "float32")
                sum_val = T.alloc_fragment((1,), "float32")

                for col in T.Parallel(block):
                    if col < N:
                        values[col] = X[row, col]
                    else:
                        values[col] = -3.402823e38

                T.reduce_max(values, max_val, dim=0, clear=True)

                for col in T.Parallel(block):
                    if col < N:
                        exps[col] = T.exp(X[row, col] - max_val[0])
                    else:
                        exps[col] = 0.0

                T.reduce_sum(exps, sum_val, dim=0, clear=True)

                for col in T.Parallel(block):
                    if col < N:
                        Y[row, col] = exps[col] / sum_val[0]

        return _kernel

    return _factory(rows, cols, block_n, dtype)


@lru_cache(maxsize=64)
def _compile_rmsnorm_parallel(rows: int, cols: int, block_n: int, dtype: str, eps: float):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(
        M: int,
        N: int,
        block: int = 256,
        dtype_name: str = "float32",
        eps_value: float = 1e-5,
    ):
        @T.prim_func
        def _kernel(
            X: T.Tensor((M, N), dtype_name),
            W: T.Tensor((N,), dtype_name),
            Y: T.Tensor((M, N), dtype_name),
        ):
            with T.Kernel(M, threads=block) as row:
                squares = T.alloc_fragment((block,), "float32")
                ss = T.alloc_fragment((1,), "float32")
                scale = T.alloc_fragment((1,), "float32")

                for col in T.Parallel(block):
                    if col < N:
                        value = X[row, col]
                        squares[col] = value * value
                    else:
                        squares[col] = 0.0

                T.reduce_sum(squares, ss, dim=0, clear=True)
                scale[0] = T.rsqrt(ss[0] / N + eps_value)

                for col in T.Parallel(block):
                    if col < N:
                        Y[row, col] = X[row, col] * scale[0] * W[col]

        return _kernel

    return _factory(rows, cols, block_n, dtype, float(eps))


@lru_cache(maxsize=64)
def _compile_scale(numel: int, block: int, dtype: str, scale: float):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(
        N: int,
        block_size: int = 256,
        dtype_name: str = "float32",
        scale_value: float = 1.0,
    ):
        @T.prim_func
        def _kernel(X: T.Tensor((N,), dtype_name), Y: T.Tensor((N,), dtype_name)):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        Y[idx] = X[idx] * scale_value

        return _kernel

    return _factory(numel, block, dtype, float(scale))


@lru_cache(maxsize=64)
def _compile_add_bias(rows: int, cols: int, block: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(
            X: T.Tensor((M, N), dtype_name),
            B: T.Tensor((N,), dtype_name),
            Y: T.Tensor((M, N), dtype_name),
        ):
            with T.Kernel(T.ceildiv(M * N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < M * N:
                        row = idx // N
                        col = idx - row * N
                        Y[row, col] = X[row, col] + B[col]

        return _kernel

    return _factory(rows, cols, block, dtype)


@lru_cache(maxsize=64)
def _compile_gelu(numel: int, block: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((N,), dtype_name), Y: T.Tensor((N,), dtype_name)):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        value = X[idx]
                        Y[idx] = 0.5 * value * (1.0 + T.erf(value * 0.7071067811865476))

        return _kernel

    return _factory(numel, block, dtype)


@lru_cache(maxsize=64)
def _compile_bias_gelu(rows: int, cols: int, block: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(
            X: T.Tensor((M, N), dtype_name),
            B: T.Tensor((N,), dtype_name),
            Y: T.Tensor((M, N), dtype_name),
        ):
            with T.Kernel(T.ceildiv(M * N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < M * N:
                        row = idx // N
                        col = idx - row * N
                        value = X[row, col] + B[col]
                        Y[row, col] = 0.5 * value * (1.0 + T.erf(value * 0.7071067811865476))

        return _kernel

    return _factory(rows, cols, block, dtype)


@lru_cache(maxsize=64)
def _compile_causal_mask(rows: int, cols: int, block: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(M: int, N: int, block_size: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((M, N), dtype_name), Y: T.Tensor((M, N), dtype_name)):
            with T.Kernel(T.ceildiv(M * N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < M * N:
                        row = idx // N
                        col = idx - row * N
                        q_pos = row - (row // N) * N
                        if col > q_pos:
                            Y[row, col] = -65504.0
                        else:
                            Y[row, col] = X[row, col]

        return _kernel

    return _factory(rows, cols, block, dtype)


@lru_cache(maxsize=64)
def _compile_scale_causal_mask(rows: int, cols: int, block: int, dtype: str, scale: float):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(
        M: int,
        N: int,
        block_size: int = 256,
        dtype_name: str = "float32",
        scale_value: float = 1.0,
    ):
        @T.prim_func
        def _kernel(X: T.Tensor((M, N), dtype_name), Y: T.Tensor((M, N), dtype_name)):
            with T.Kernel(T.ceildiv(M * N, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < M * N:
                        row = idx // N
                        col = idx - row * N
                        q_pos = row - (row // N) * N
                        if col > q_pos:
                            Y[row, col] = -65504.0
                        else:
                            Y[row, col] = X[row, col] * scale_value

        return _kernel

    return _factory(rows, cols, block, dtype, float(scale))


def gemm_tilelang(
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
        raise ValueError("a and b must be rank-2")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    _require_gemm_tile_aligned(a.shape[0], b.shape[1], a.shape[1], block_m, block_n, block_k)
    require_cuda_contiguous(a, b)
    kernel = _compile_gemm(
        a.shape[0],
        b.shape[1],
        a.shape[1],
        block_m,
        block_n,
        block_k,
        threads,
        num_stages,
        dtype_name(a.dtype),
        accum_dtype,
    )
    return kernel(a, b)


def row_softmax_tilelang(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    require_cuda_contiguous(x)
    out = torch.empty_like(x)
    kernel = _compile_row_softmax(x.shape[0], x.shape[1], dtype_name(x.dtype))
    kernel(x, out)
    return out


def row_softmax_parallel_tilelang(x: torch.Tensor, block_n: int = 256) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    require_single_block_reduction("row_softmax_parallel_tilelang", x.shape[1], block_n)
    require_cuda_contiguous(x)
    out = torch.empty_like(x)
    kernel = _compile_row_softmax_parallel(x.shape[0], x.shape[1], block_n, dtype_name(x.dtype))
    kernel(x, out)
    return out


def rmsnorm_tilelang(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must have shape (N,)")
    require_cuda_contiguous(x, weight)
    out = torch.empty_like(x)
    kernel = _compile_rmsnorm(x.shape[0], x.shape[1], dtype_name(x.dtype), float(eps))
    kernel(x, weight, out)
    return out


def rmsnorm_parallel_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    block_n: int = 256,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must have shape (N,)")
    require_single_block_reduction("rmsnorm_parallel_tilelang", x.shape[1], block_n)
    require_cuda_contiguous(x, weight)
    out = torch.empty_like(x)
    kernel = _compile_rmsnorm_parallel(x.shape[0], x.shape[1], block_n, dtype_name(x.dtype), float(eps))
    kernel(x, weight, out)
    return out


def gelu_tilelang(x: torch.Tensor, block: int = 256) -> torch.Tensor:
    require_cuda_contiguous(x)
    out = torch.empty_like(x)
    kernel = _compile_gelu(x.numel(), block, dtype_name(x.dtype))
    kernel(x.reshape(-1), out.reshape(-1))
    return out


def scale_tilelang(x: torch.Tensor, scale: float, block: int = 256) -> torch.Tensor:
    require_cuda_contiguous(x)
    out = torch.empty_like(x)
    kernel = _compile_scale(x.numel(), block, dtype_name(x.dtype), float(scale))
    kernel(x.reshape(-1), out.reshape(-1))
    return out


def add_bias_tilelang(x: torch.Tensor, bias: torch.Tensor, block: int = 256) -> torch.Tensor:
    if x.ndim != 2 or bias.ndim != 1:
        raise ValueError("x must be rank-2 and bias must be rank-1")
    if x.shape[1] != bias.shape[0]:
        raise ValueError("expected x=(M,N), bias=(N,)")
    require_cuda_contiguous(x, bias)
    out = torch.empty_like(x)
    kernel = _compile_add_bias(x.shape[0], x.shape[1], block, dtype_name(x.dtype))
    kernel(x, bias, out)
    return out


def linear_bias_tilelang(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("x and weight must be rank-2, bias must be rank-1")
    if x.shape[1] != weight.shape[0] or weight.shape[1] != bias.shape[0]:
        raise ValueError("expected x=(M,K), weight=(K,N), bias=(N,)")
    require_cuda_contiguous(x, weight, bias)
    return add_bias_tilelang(gemm_tilelang(x, weight), bias)


def causal_mask_tilelang(scores: torch.Tensor, seq_len: int, block: int = 256) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2")
    if scores.shape[1] != seq_len:
        raise ValueError("scores.shape[1] must equal seq_len")
    require_cuda_contiguous(scores)
    out = torch.empty_like(scores)
    kernel = _compile_causal_mask(scores.shape[0], scores.shape[1], block, dtype_name(scores.dtype))
    kernel(scores, out)
    return out


def scale_causal_mask_tilelang(
    scores: torch.Tensor,
    scale: float,
    seq_len: int,
    block: int = 256,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be rank-2")
    if scores.shape[1] != seq_len:
        raise ValueError("scores.shape[1] must equal seq_len")
    require_cuda_contiguous(scores)
    out = torch.empty_like(scores)
    kernel = _compile_scale_causal_mask(
        scores.shape[0],
        scores.shape[1],
        block,
        dtype_name(scores.dtype),
        float(scale),
    )
    kernel(scores, out)
    return out


def linear_bias_gelu_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    block: int = 256,
) -> torch.Tensor:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("x and weight must be rank-2, bias must be rank-1")
    if x.shape[1] != weight.shape[0] or weight.shape[1] != bias.shape[0]:
        raise ValueError("expected x=(M,K), weight=(K,N), bias=(N,)")
    require_cuda_contiguous(x, weight, bias)
    hidden = gemm_tilelang(x, weight)
    out = torch.empty_like(hidden)
    kernel = _compile_bias_gelu(hidden.shape[0], hidden.shape[1], block, dtype_name(hidden.dtype))
    kernel(hidden, bias, out)
    return out
