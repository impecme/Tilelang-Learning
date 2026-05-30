from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tilelab.advanced import (
    gemm_reference,
    gemm_tilelang,
    rmsnorm_parallel_tilelang,
    rmsnorm_reference,
    row_softmax_parallel_tilelang,
    row_softmax_reference,
)
from tilelab.basic import row_sum_parallel_tilelang, row_sum_reference, vector_add_reference, vector_add_tilelang
from tilelab.common import BenchmarkResult, write_benchmark_csv
from tilelab.decoder import (
    MiniDecoderConfig,
    decoder_block_reference,
    decoder_block_tilelang_optimized,
    make_random_weights,
    mini_inference_reference,
)


LabFn = Callable[[bool], None]


def _require_cuda(run_tilelang: bool) -> None:
    if run_tilelang and not torch.cuda.is_available():
        raise RuntimeError("--run-tilelang requires CUDA")


def lab_basic(run_tilelang: bool) -> None:
    torch.manual_seed(10)
    a = torch.randn(1000)
    b = torch.randn(1000)
    torch.testing.assert_close(vector_add_reference(a, b), a + b)
    x = torch.randn((8, 17), dtype=torch.float32)
    torch.testing.assert_close(row_sum_reference(x), x.sum(dim=-1))
    print("[basic] reference vector_add and row_sum ok")

    if run_tilelang:
        _require_cuda(True)
        torch.testing.assert_close(vector_add_tilelang(a.cuda(), b.cuda()).cpu(), a + b)
        torch.testing.assert_close(row_sum_parallel_tilelang(x.cuda(), block_n=32).cpu(), x.sum(dim=-1))
        print("[basic] tilelang vector_add and row_sum_parallel ok")


def lab_gemm(run_tilelang: bool) -> None:
    torch.manual_seed(11)
    a = torch.randn((128, 128), dtype=torch.float16)
    b = torch.randn((128, 128), dtype=torch.float16)
    expected = gemm_reference(a, b)
    print(f"[gemm] reference output shape={tuple(expected.shape)}")

    try:
        gemm_tilelang(torch.randn((96, 128), dtype=torch.float16), b)
    except ValueError as exc:
        print(f"[gemm] expected tile-aligned error: {exc}")

    if run_tilelang:
        _require_cuda(True)
        actual = gemm_tilelang(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print("[gemm] tilelang GEMM ok")


def lab_reduction(run_tilelang: bool) -> None:
    torch.manual_seed(12)
    x = torch.randn((8, 16), dtype=torch.float32)
    w = torch.randn((16,), dtype=torch.float32)
    torch.testing.assert_close(row_softmax_reference(x), torch.softmax(x, dim=-1))
    torch.testing.assert_close(rmsnorm_reference(x, w), rmsnorm_reference(x, w))
    print("[reduction] reference softmax and rmsnorm ok")

    try:
        row_sum_parallel_tilelang(x, block_n=8)
    except ValueError as exc:
        print(f"[reduction] expected cols > block_n error: {exc}")

    if run_tilelang:
        _require_cuda(True)
        x_cuda = x.cuda()
        w_cuda = w.cuda()
        torch.testing.assert_close(row_sum_parallel_tilelang(x_cuda, block_n=16).cpu(), row_sum_reference(x))
        torch.testing.assert_close(row_softmax_parallel_tilelang(x_cuda, block_n=16).cpu(), row_softmax_reference(x))
        torch.testing.assert_close(rmsnorm_parallel_tilelang(x_cuda, w_cuda, block_n=16).cpu(), rmsnorm_reference(x, w))
        print("[reduction] tilelang parallel reductions ok")


def lab_decoder(run_tilelang: bool) -> None:
    dtype_name = "float16" if run_tilelang else "float32"
    device = "cuda" if run_tilelang else "cpu"
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype=dtype_name,
    )
    torch.manual_seed(13)
    weights = make_random_weights(config, seed=14, device=device)
    x = torch.randn((1, config.seq_len, config.hidden_size), device=device, dtype=getattr(torch, dtype_name))
    block_out = decoder_block_reference(x, weights, config)
    logits = mini_inference_reference(x, weights, config)
    print(f"[decoder] block_out={tuple(block_out.shape)}, logits={tuple(logits.shape)}")

    if run_tilelang:
        _require_cuda(True)
        actual = decoder_block_tilelang_optimized(x, weights, config)
        torch.testing.assert_close(actual, block_out, rtol=3e-2, atol=3e-2)
        print("[decoder] optimized TileLang block close to reference")


def lab_benchmark(run_tilelang: bool) -> None:
    result = BenchmarkResult(
        "example_latency",
        0.123456,
        1,
        2,
        suite="lab",
        shape="M=8,N=16",
        dtype="float32",
        backend="example",
        notes="teaching row, not a real benchmark",
    )
    tmp_path = Path("/tmp/tilelang_lab_example.csv")
    write_benchmark_csv(tmp_path, [result])
    print(f"[benchmark] wrote example CSV: {tmp_path}")
    print("[benchmark] fields: suite,name,backend,shape,dtype,latency_ms,warmup,repeat,notes")
    if run_tilelang:
        print("[benchmark] run full benchmark separately: python3 -m benchmarks.bench_reductions --run-tilelang ...")


LABS: dict[str, tuple[str, LabFn]] = {
    "basic": ("Basic Kernel Debug", lab_basic),
    "gemm": ("GEMM Tile Shape", lab_gemm),
    "reduction": ("Reduction Optimization", lab_reduction),
    "decoder": ("Decoder Shape Trace", lab_decoder),
    "benchmark": ("Benchmark Reading", lab_benchmark),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list available labs")
    parser.add_argument("--lab", choices=("all",) + tuple(LABS), default="all")
    parser.add_argument("--run-tilelang", action="store_true", help="also run CUDA TileLang kernels")
    args = parser.parse_args()

    if args.list:
        print("== Full Labs ==")
        for lab_id, (name, _) in LABS.items():
            print(f"{lab_id}: {name}")
        return 0

    selected = LABS.items() if args.lab == "all" else [(args.lab, LABS[args.lab])]
    for lab_id, (name, fn) in selected:
        print(f"\n== Lab {lab_id}: {name} ==")
        fn(args.run_tilelang)
    print("\nfull labs ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
