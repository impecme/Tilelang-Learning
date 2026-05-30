"""Shared helpers for the TileLang learning lab."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


SUPPORTED_DTYPES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def dtype_name(dtype: torch.dtype) -> str:
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported dtype: {dtype}")
    return SUPPORTED_DTYPES[dtype]


def require_same_shape(*tensors: torch.Tensor) -> None:
    if not tensors:
        return
    shape = tensors[0].shape
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError("all tensors must have the same shape")


def require_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("TileLang kernels expect CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("TileLang kernels expect contiguous tensors")


def require_single_block_reduction(op_name: str, cols: int, block_n: int) -> None:
    if block_n <= 0:
        raise ValueError(f"{op_name} requires block_n to be positive, got {block_n}")
    if block_n & (block_n - 1):
        raise ValueError(f"{op_name} requires block_n to be a power of two, got {block_n}")
    if block_n > 1024:
        raise ValueError(f"{op_name} requires block_n <= 1024, got {block_n}")
    if cols > block_n:
        raise ValueError(
            f"{op_name} v1 supports one row per CUDA block, so cols must be <= block_n; "
            f"got cols={cols}, block_n={block_n}"
        )


def load_tilelang():
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:  # pragma: no cover - exercised by optional smoke tests
        raise RuntimeError("tilelang is not installed or failed to import") from exc
    return tilelang, T


def gelu_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    latency_ms: float
    warmup: int
    repeat: int
    suite: str = ""
    shape: str = ""
    dtype: str = ""
    backend: str = ""
    notes: str = ""


def cuda_time_ms(fn: Callable[[], object], warmup: int = 10, repeat: int = 25) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark timing")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.name}: {result.latency_ms:.4f} ms "
        f"(warmup={result.warmup}, repeat={result.repeat})"
    )


CSV_FIELDS = [
    "suite",
    "name",
    "backend",
    "shape",
    "dtype",
    "latency_ms",
    "warmup",
    "repeat",
    "notes",
]


def benchmark_row(result: BenchmarkResult) -> dict[str, object]:
    return {
        "suite": result.suite,
        "name": result.name,
        "backend": result.backend,
        "shape": result.shape,
        "dtype": result.dtype,
        "latency_ms": f"{result.latency_ms:.6f}",
        "warmup": result.warmup,
        "repeat": result.repeat,
        "notes": result.notes,
    }


def write_benchmark_csv(path: str | Path, results: list[BenchmarkResult]) -> None:
    csv_path = Path(path)
    if csv_path.parent != Path("."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(benchmark_row(result))
