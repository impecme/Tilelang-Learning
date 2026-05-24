"""Benchmark the capstone FlashAttention interface against PyTorch baselines."""

from __future__ import annotations

import argparse

import torch

from benchmarks.utils import BenchmarkResult, cuda_time_ms, print_result
from kernels.flash_attention import flash_attention_forward
from kernels.reference import naive_attention_forward, sdpa_attention_forward


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = _dtype(args.dtype)
    shape = (args.batch, args.heads, args.seq, args.dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    print(f"shape={shape}, dtype={dtype}, causal={args.causal}")
    print_result(
        BenchmarkResult(
            "project_flash_attention_api",
            cuda_time_ms(
                lambda: flash_attention_forward(q, k, v, causal=args.causal),
                args.warmup,
                args.repeat,
            ),
            args.warmup,
            args.repeat,
        )
    )
    print_result(
        BenchmarkResult(
            "torch_sdpa",
            cuda_time_ms(
                lambda: sdpa_attention_forward(q, k, v, causal=args.causal),
                args.warmup,
                args.repeat,
            ),
            args.warmup,
            args.repeat,
        )
    )

    if args.seq <= 1024:
        print_result(
            BenchmarkResult(
                "naive_materialized_attention",
                cuda_time_ms(
                    lambda: naive_attention_forward(q, k, v, causal=args.causal),
                    args.warmup,
                    args.repeat,
                ),
                args.warmup,
                args.repeat,
            )
        )


if __name__ == "__main__":
    main()

