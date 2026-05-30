from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter_tilelab import (
    TinyModelConfig,
    gemm_reference,
    gemm_tilelang,
    make_tiny_weights,
    tiny_model_reference,
    tiny_model_tilelang,
    vector_add_reference,
    vector_add_tilelang,
)


LabFn = Callable[[bool], None]


def _device_for_tilelang(run_tilelang: bool) -> str:
    if run_tilelang and not torch.cuda.is_available():
        raise RuntimeError("--run-tilelang requires CUDA")
    return "cuda" if run_tilelang else "cpu"


def lab_vector_add(run_tilelang: bool) -> None:
    torch.manual_seed(1)
    a = torch.randn(1000)
    b = torch.randn(1000)
    expected = vector_add_reference(a, b)
    torch.testing.assert_close(expected, a + b)
    print("[lab 1] reference vector_add N=1000 ok")

    if run_tilelang:
        actual = vector_add_tilelang(a.cuda(), b.cuda(), block_size=128).cpu()
        torch.testing.assert_close(actual, expected)
        print("[lab 1] tilelang vector_add block_size=128 ok")


def lab_gemm(run_tilelang: bool) -> None:
    torch.manual_seed(2)
    a = torch.randn((16, 32), dtype=torch.float16)
    b = torch.randn((32, 16), dtype=torch.float16)
    expected = gemm_reference(a, b)
    torch.testing.assert_close(expected, a @ b)
    print("[lab 2] reference GEMM A=(16,32), B=(32,16) ok")

    try:
        bad_a = torch.randn((15, 32), dtype=torch.float16)
        gemm_tilelang(bad_a, b)
    except ValueError as exc:
        print(f"[lab 2] expected tile-aligned error: {exc}")

    if run_tilelang:
        actual = gemm_tilelang(a.cuda(), b.cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print("[lab 2] tilelang GEMM ok")


def lab_tiny_model(run_tilelang: bool) -> None:
    device = _device_for_tilelang(run_tilelang)
    dtype_name = "float16" if run_tilelang else "float32"
    config = TinyModelConfig(
        seq_len=16,
        hidden_size=64,
        ffn_hidden_size=128,
        vocab_size=256,
        dtype=dtype_name,
    )
    torch.manual_seed(3)
    weights = make_tiny_weights(config, seed=4, device=device)
    x = torch.randn((config.seq_len, config.hidden_size), device=device, dtype=getattr(torch, dtype_name))
    expected = tiny_model_reference(x, weights, config)
    print(f"[lab 3] reference tiny model logits shape={tuple(expected.shape)}")

    if run_tilelang:
        actual = tiny_model_tilelang(x, weights, config)
        torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
        print("[lab 3] tilelang tiny model logits close to reference")


LABS: dict[str, tuple[str, LabFn]] = {
    "1": ("Vector Add", lab_vector_add),
    "2": ("GEMM", lab_gemm),
    "3": ("Tiny Model Flow", lab_tiny_model),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list available starter labs")
    parser.add_argument("--lab", choices=("all",) + tuple(LABS), default="all")
    parser.add_argument("--run-tilelang", action="store_true", help="also run CUDA TileLang kernels")
    args = parser.parse_args()

    if args.list:
        print("== Starter Labs ==")
        for lab_id, (name, _) in LABS.items():
            print(f"{lab_id}: {name}")
        return 0

    selected = LABS.items() if args.lab == "all" else [(args.lab, LABS[args.lab])]
    for lab_id, (name, fn) in selected:
        print(f"\n== Lab {lab_id}: {name} ==")
        fn(args.run_tilelang)
    print("\nstarter labs ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
