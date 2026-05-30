from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tilelab.advanced import gemm_tilelang
from tilelab.basic import row_sum_parallel_tilelang, vector_add_tilelang
from tilelab.decoder import MiniDecoderConfig, decoder_block_reference, make_random_weights


CaseFn = Callable[[], None]


def _show(case_id: str, title: str, why: str, fix: str, doc: str, fn: CaseFn) -> None:
    print(f"\n== {case_id}: {title} ==")
    try:
        fn()
    except Exception as exc:
        print(f"caught: {type(exc).__name__}: {exc}")
        print(f"why: {why}")
        print(f"fix: {fix}")
        print(f"doc: {doc}")
        return
    print("warning: this case did not raise; check the demo setup")


def case_cpu_tensor() -> None:
    a = torch.randn(8)
    b = torch.randn(8)
    vector_add_tilelang(a, b)


def case_non_contiguous() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to demonstrate a non-contiguous CUDA tensor")
    x = torch.randn((4, 8), device="cuda").t()
    y = torch.randn_like(x)
    vector_add_tilelang(x, y)


def case_gemm_shape() -> None:
    a = torch.randn((96, 128), dtype=torch.float16)
    b = torch.randn((128, 128), dtype=torch.float16)
    gemm_tilelang(a, b)


def case_reduction_block() -> None:
    x = torch.randn((4, 300), dtype=torch.float32)
    row_sum_parallel_tilelang(x, block_n=256)


def _decoder_config() -> MiniDecoderConfig:
    return MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float32",
    )


def case_decoder_shape() -> None:
    config = _decoder_config()
    weights = make_random_weights(config, seed=1, device="cpu")
    bad_weights = replace(weights, qkv_weight=torch.randn((127, 384), dtype=torch.float32))
    x = torch.randn((1, 128, 128), dtype=torch.float32)
    decoder_block_reference(x, bad_weights, config)


def case_decoder_dtype() -> None:
    config = _decoder_config()
    weights = make_random_weights(config, seed=2, device="cpu")
    bad_weights = replace(weights, norm1_weight=weights.norm1_weight.half())
    x = torch.randn((1, 128, 128), dtype=torch.float32)
    decoder_block_reference(x, bad_weights, config)


def case_decoder_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to demonstrate a CPU/GPU device mismatch")
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float16",
    )
    weights = make_random_weights(config, seed=3, device="cpu")
    x = torch.randn((1, 128, 128), device="cuda", dtype=torch.float16)
    decoder_block_reference(x, weights, config)


CASES: dict[str, tuple[str, str, str, str, CaseFn]] = {
    "cpu_tensor": (
        "CPU tensor passed to TileLang",
        "TileLang kernels expect CUDA tensors, but the input lives on CPU.",
        "Use PyTorch reference on CPU, or move tensors to CUDA before calling TileLang.",
        "docs/error_gallery.md#cpu-tensor",
        case_cpu_tensor,
    ),
    "non_contiguous": (
        "Non-contiguous CUDA tensor",
        "Transpose/permute can create a tensor whose memory layout is not contiguous.",
        "Call .contiguous() before passing the tensor to TileLang.",
        "docs/error_gallery.md#non-contiguous-tensor",
        case_non_contiguous,
    ),
    "gemm_shape": (
        "GEMM shape is not tile-aligned",
        "The teaching GEMM does not implement tail-tile guards.",
        "Use M/N/K divisible by block_m/block_n/block_k.",
        "docs/kernels/gemm.md",
        case_gemm_shape,
    ),
    "reduction_block": (
        "Reduction row is wider than block_n",
        "Parallel reduction v1 only supports one row inside one CUDA block.",
        "Use cols <= block_n <= 1024 or fall back to the serial teaching kernel.",
        "docs/reduction_optimization.md",
        case_reduction_block,
    ),
    "decoder_shape": (
        "Decoder weight shape mismatch",
        "Decoder weights must exactly match MiniDecoderConfig.",
        "Create weights with make_random_weights or fix the reported shape.",
        "docs/shape_walkthroughs.md",
        case_decoder_shape,
    ),
    "decoder_dtype": (
        "Decoder weight dtype mismatch",
        "All weights must use the dtype requested by MiniDecoderConfig.",
        "Regenerate weights with the same config dtype.",
        "docs/error_gallery.md#decoder-权重-shapedtypedevice",
        case_decoder_dtype,
    ),
    "decoder_device": (
        "Decoder CPU/GPU device mismatch",
        "Input and all weights must be on the same device.",
        "Move weights and x to the same device.",
        "docs/error_gallery.md#decoder-权重-shapedtypedevice",
        case_decoder_device,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all",) + tuple(CASES), default="all")
    parser.add_argument("--list", action="store_true", help="list available demos")
    args = parser.parse_args()

    if args.list:
        print("== Common Error Demos ==")
        for case_id, (title, *_rest) in CASES.items():
            print(f"{case_id}: {title}")
        return 0

    selected = CASES.items() if args.case == "all" else [(args.case, CASES[args.case])]
    for case_id, (title, why, fix, doc, fn) in selected:
        _show(case_id, title, why, fix, doc, fn)
    print("\nerror demos completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
