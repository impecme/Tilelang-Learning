from __future__ import annotations

import argparse

import torch

from tilelab.basic import axpy_reference, axpy_tilelang, vector_add_reference, vector_add_tilelang
from tilelab.common import BenchmarkResult, cuda_time_ms, print_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--run-tilelang", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    a = torch.randn((args.numel,), device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    print(f"numel={args.numel}, dtype=float32")
    print_result(
        BenchmarkResult(
            "torch_vector_add",
            cuda_time_ms(lambda: vector_add_reference(a, b), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
        )
    )
    print_result(
        BenchmarkResult(
            "torch_axpy",
            cuda_time_ms(lambda: axpy_reference(1.25, a, b), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
        )
    )

    if args.run_tilelang:
        print_result(
            BenchmarkResult(
                "tilelang_vector_add",
                cuda_time_ms(lambda: vector_add_tilelang(a, b), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
            )
        )
        print_result(
            BenchmarkResult(
                "tilelang_axpy",
                cuda_time_ms(lambda: axpy_tilelang(1.25, a, b), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
            )
        )


if __name__ == "__main__":
    main()

