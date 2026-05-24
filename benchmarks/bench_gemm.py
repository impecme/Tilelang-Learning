"""Benchmark TileLang GEMM against torch matmul."""

from __future__ import annotations

import argparse

import torch

from benchmarks.utils import BenchmarkResult, cuda_time_ms, print_result
from kernels.gemm import matmul_reference, matmul_tilelang


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--run-tilelang", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)

    print(f"shape=({args.m}, {args.n}, {args.k}), dtype=float16")
    print_result(
        BenchmarkResult(
            "torch_matmul",
            cuda_time_ms(lambda: matmul_reference(a, b), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
        )
    )

    if args.run_tilelang:
        print_result(
            BenchmarkResult(
                "tilelang_gemm",
                cuda_time_ms(lambda: matmul_tilelang(a, b), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
            )
        )


if __name__ == "__main__":
    main()
