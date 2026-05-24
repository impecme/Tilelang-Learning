"""Small CUDA benchmark utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    latency_ms: float
    warmup: int
    repeat: int


def cuda_time_ms(fn: Callable[[], object], warmup: int = 25, repeat: int = 100) -> float:
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

