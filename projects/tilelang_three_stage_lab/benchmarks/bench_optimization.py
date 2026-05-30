from __future__ import annotations

import argparse

import torch

from tilelab.advanced import (
    add_bias_tilelang,
    linear_bias_tilelang,
    scale_causal_mask_tilelang,
    scale_tilelang,
)
from tilelab.common import BenchmarkResult, cuda_time_ms, print_result, write_benchmark_csv


def record(result: BenchmarkResult, results: list[BenchmarkResult]) -> None:
    results.append(result)
    print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--run-tilelang", action="store_true")
    parser.add_argument("--csv", type=str, default="", help="optional path to write benchmark CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    scale = 0.125
    x = torch.randn((args.m, args.n), device="cuda", dtype=torch.float16)
    bias = torch.randn((args.n,), device="cuda", dtype=torch.float16)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    weight = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    scores = torch.randn((args.seq, args.seq), device="cuda", dtype=torch.float16)
    row = torch.arange(args.seq, device="cuda").view(args.seq, 1)
    col = torch.arange(args.seq, device="cuda").view(1, args.seq)
    causal_mask = col > row

    results: list[BenchmarkResult] = []
    matrix_shape = f"M={args.m},N={args.n}"
    gemm_shape = f"M={args.m},N={args.n},K={args.k}"
    score_shape = f"S={args.seq},S={args.seq}"

    print(
        f"optimization matrix=({args.m},{args.n}), gemm=({args.m},{args.n},{args.k}), "
        f"scores=({args.seq},{args.seq}), dtype=float16"
    )
    record(
        BenchmarkResult(
            "torch_scale",
            cuda_time_ms(lambda: x * scale, args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="optimization",
            shape=matrix_shape,
            dtype="float16",
            backend="torch",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_add_bias",
            cuda_time_ms(lambda: x + bias, args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="optimization",
            shape=matrix_shape,
            dtype="float16",
            backend="torch",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_scale_causal_mask",
            cuda_time_ms(lambda: (scores * scale).masked_fill(causal_mask, -65504.0), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="optimization",
            shape=score_shape,
            dtype="float16",
            backend="torch",
            notes="mask tensor is precomputed",
        ),
        results,
    )
    record(
        BenchmarkResult(
            "torch_linear_bias",
            cuda_time_ms(lambda: a @ weight + bias, args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="optimization",
            shape=gemm_shape,
            dtype="float16",
            backend="torch",
        ),
        results,
    )

    if args.run_tilelang:
        record(
            BenchmarkResult(
                "tilelang_scale",
                cuda_time_ms(lambda: scale_tilelang(x, scale), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="optimization",
                shape=matrix_shape,
                dtype="float16",
                backend="tilelang",
                notes="single teaching elementwise kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_add_bias",
                cuda_time_ms(lambda: add_bias_tilelang(x, bias), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="optimization",
                shape=matrix_shape,
                dtype="float16",
                backend="tilelang",
                notes="single teaching row-bias kernel",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_scale_causal_mask",
                cuda_time_ms(lambda: scale_causal_mask_tilelang(scores, scale, args.seq), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="optimization",
                shape=score_shape,
                dtype="float16",
                backend="tilelang",
                notes="fuses score scale and causal mask",
            ),
            results,
        )
        record(
            BenchmarkResult(
                "tilelang_linear_bias",
                cuda_time_ms(lambda: linear_bias_tilelang(a, weight, bias), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="optimization",
                shape=gemm_shape,
                dtype="float16",
                backend="tilelang",
                notes="teaching composition: gemm_tilelang + add_bias_tilelang",
            ),
            results,
        )

    if args.csv:
        write_benchmark_csv(args.csv, results)
        print(f"wrote csv: {args.csv}")


if __name__ == "__main__":
    main()
