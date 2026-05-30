from __future__ import annotations

import argparse

import torch

from tilelab.advanced import (
    gelu_reference,
    gelu_tilelang,
    gemm_reference,
    gemm_tilelang,
    rmsnorm_reference,
    rmsnorm_tilelang,
)
from tilelab.common import BenchmarkResult, cuda_time_ms, print_result
from tilelab.common import write_benchmark_csv


def record(result: BenchmarkResult, results: list[BenchmarkResult]) -> None:
    results.append(result)
    print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--run-tilelang", action="store_true")
    parser.add_argument("--csv", type=str, default="", help="optional path to write benchmark CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    x = torch.randn((args.m, args.n), device="cuda", dtype=torch.float16)
    w = torch.randn((args.n,), device="cuda", dtype=torch.float16)
    results: list[BenchmarkResult] = []
    gemm_shape = f"M={args.m},N={args.n},K={args.k}"
    matrix_shape = f"M={args.m},N={args.n}"

    print(f"gemm=({args.m},{args.n},{args.k}), op_matrix=({args.m},{args.n}), dtype=float16")
    record(
        BenchmarkResult(
            "torch_gemm",
            cuda_time_ms(lambda: gemm_reference(a, b), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="advanced",
            shape=gemm_shape,
            dtype="float16",
            backend="torch",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_rmsnorm",
            cuda_time_ms(lambda: rmsnorm_reference(x, w), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="advanced",
            shape=matrix_shape,
            dtype="float16",
            backend="torch",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_gelu",
            cuda_time_ms(lambda: gelu_reference(x), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="advanced",
            shape=matrix_shape,
            dtype="float16",
            backend="torch",
        ),
        results,
    )

    if args.run_tilelang:
        record(
            BenchmarkResult(
                "tilelang_gemm",
                cuda_time_ms(lambda: gemm_tilelang(a, b), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="advanced",
                shape=gemm_shape,
                dtype="float16",
                backend="tilelang",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_rmsnorm",
                cuda_time_ms(lambda: rmsnorm_tilelang(x, w), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="advanced",
                shape=matrix_shape,
                dtype="float16",
                backend="tilelang",
                notes="serial reduction teaching kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_gelu",
                cuda_time_ms(lambda: gelu_tilelang(x), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="advanced",
                shape=matrix_shape,
                dtype="float16",
                backend="tilelang",
            ),
            results,
        )

    if args.csv:
        write_benchmark_csv(args.csv, results)
        print(f"wrote csv: {args.csv}")


if __name__ == "__main__":
    main()
