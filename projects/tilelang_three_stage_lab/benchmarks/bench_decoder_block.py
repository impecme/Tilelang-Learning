from __future__ import annotations

import argparse

import torch

from tilelab.common import BenchmarkResult, cuda_time_ms, print_result, write_benchmark_csv
from tilelab.decoder import (
    MiniDecoderConfig,
    decoder_block_reference,
    decoder_block_tilelang,
    decoder_block_tilelang_optimized,
    make_random_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--run-tilelang", action="store_true")
    parser.add_argument("--compare-optimized", action="store_true")
    parser.add_argument("--csv", type=str, default="", help="optional path to write benchmark CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = MiniDecoderConfig(
        hidden_size=args.hidden,
        num_heads=args.heads,
        head_dim=args.hidden // args.heads,
        ffn_hidden_size=args.ffn,
        vocab_size=args.vocab,
        seq_len=args.seq,
        dtype="float16",
    )
    weights = make_random_weights(config, seed=0, device="cuda")
    x = torch.randn((1, args.seq, args.hidden), device="cuda", dtype=torch.float16)
    results: list[BenchmarkResult] = []
    shape = f"B=1,S={args.seq},H={args.hidden},heads={args.heads},ffn={args.ffn},vocab={args.vocab}"

    print(f"decoder_block=(B=1,S={args.seq},H={args.hidden},heads={args.heads}), dtype=float16")
    result = BenchmarkResult(
        "torch_decoder_block_reference",
        cuda_time_ms(lambda: decoder_block_reference(x, weights, config), args.warmup, args.repeat),
        args.warmup,
        args.repeat,
        suite="decoder_block",
        shape=shape,
        dtype="float16",
        backend="torch",
    )
    results.append(result)
    print_result(result)

    if args.run_tilelang:
        result = BenchmarkResult(
            "tilelang_decoder_block",
            cuda_time_ms(lambda: decoder_block_tilelang(x, weights, config), args.warmup, args.repeat),
            args.warmup,
            args.repeat,
            suite="decoder_block",
            shape=shape,
            dtype="float16",
            backend="tilelang",
            notes="uses TileLang utility kernels for scale, bias, residual, mask",
        )
        results.append(result)
        print_result(result)

        if args.compare_optimized:
            result = BenchmarkResult(
                "tilelang_decoder_block_optimized_reductions",
                cuda_time_ms(lambda: decoder_block_tilelang_optimized(x, weights, config), args.warmup, args.repeat),
                args.warmup,
                args.repeat,
                suite="decoder_block",
                shape=shape,
                dtype="float16",
                backend="tilelang",
                notes="uses parallel reduction kernels for RMSNorm and attention softmax",
            )
            results.append(result)
            print_result(result)

    if args.csv:
        write_benchmark_csv(args.csv, results)
        print(f"wrote csv: {args.csv}")


if __name__ == "__main__":
    main()
