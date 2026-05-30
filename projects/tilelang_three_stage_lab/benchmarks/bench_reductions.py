from __future__ import annotations

import argparse

import torch

from tilelab.advanced import (
    rmsnorm_parallel_tilelang,
    rmsnorm_reference,
    rmsnorm_tilelang,
    row_softmax_parallel_tilelang,
    row_softmax_reference,
    row_softmax_tilelang,
)
from tilelab.basic import row_sum_parallel_tilelang, row_sum_reference, row_sum_tilelang
from tilelab.common import BenchmarkResult, cuda_time_ms, print_result, write_benchmark_csv


def record(result: BenchmarkResult, results: list[BenchmarkResult]) -> None:
    results.append(result)
    print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--run-tilelang", action="store_true")
    parser.add_argument("--csv", type=str, default="", help="optional path to write benchmark CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = getattr(torch, args.dtype)
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=dtype)
    weight = torch.randn((args.cols,), device="cuda", dtype=dtype)
    results: list[BenchmarkResult] = []
    shape = f"M={args.rows},N={args.cols},block_n={args.block_n}"

    print(f"reductions=({args.rows},{args.cols}), block_n={args.block_n}, dtype={args.dtype}")
    record(
        BenchmarkResult(
            "torch_row_sum",
            cuda_time_ms(lambda: row_sum_reference(x), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="reductions",
            shape=shape,
            dtype=args.dtype,
            backend="torch",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_row_softmax",
            cuda_time_ms(lambda: row_softmax_reference(x), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="reductions",
            shape=shape,
            dtype=args.dtype,
            backend="torch",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_rmsnorm",
            cuda_time_ms(lambda: rmsnorm_reference(x, weight), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="reductions",
            shape=shape,
            dtype=args.dtype,
            backend="torch",
        ),
        results,
    )

    if args.run_tilelang:
        record(
            BenchmarkResult(
                "tilelang_row_sum_serial",
                cuda_time_ms(lambda: row_sum_tilelang(x), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="reductions",
                shape=shape,
                dtype=args.dtype,
                backend="tilelang",
                notes="serial reduction teaching kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_row_sum_parallel",
                cuda_time_ms(lambda: row_sum_parallel_tilelang(x, block_n=args.block_n), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="reductions",
                shape=shape,
                dtype=args.dtype,
                backend="tilelang",
                notes="single-block T.reduce_sum teaching kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_row_softmax_serial",
                cuda_time_ms(lambda: row_softmax_tilelang(x), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="reductions",
                shape=shape,
                dtype=args.dtype,
                backend="tilelang",
                notes="serial reduction teaching kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_row_softmax_parallel",
                cuda_time_ms(lambda: row_softmax_parallel_tilelang(x, block_n=args.block_n), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="reductions",
                shape=shape,
                dtype=args.dtype,
                backend="tilelang",
                notes="single-block T.reduce_max + T.reduce_sum teaching kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_rmsnorm_serial",
                cuda_time_ms(lambda: rmsnorm_tilelang(x, weight), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="reductions",
                shape=shape,
                dtype=args.dtype,
                backend="tilelang",
                notes="serial reduction teaching kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_rmsnorm_parallel",
                cuda_time_ms(lambda: rmsnorm_parallel_tilelang(x, weight, block_n=args.block_n), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="reductions",
                shape=shape,
                dtype=args.dtype,
                backend="tilelang",
                notes="single-block T.reduce_sum teaching kernel",
            ),
            results,
        )

    if args.csv:
        write_benchmark_csv(args.csv, results)
        print(f"wrote csv: {args.csv}")


if __name__ == "__main__":
    main()
