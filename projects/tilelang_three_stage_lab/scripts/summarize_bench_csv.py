from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = {
    "suite",
    "name",
    "backend",
    "shape",
    "dtype",
    "latency_ms",
    "warmup",
    "repeat",
    "notes",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required fields: {sorted(missing)}")
        rows = list(reader)
    for row in rows:
        try:
            float(row["latency_ms"])
        except ValueError as exc:
            raise ValueError(f"invalid latency_ms in row {row}") from exc
    return rows


def latency(row: dict[str, str]) -> float:
    return float(row["latency_ms"])


def short(row: dict[str, str]) -> str:
    return f"{row['name']} [{row['backend']}] {latency(row):.4f} ms"


def normalize_op_name(name: str) -> str:
    base = name
    for prefix in ("torch_", "tilelang_"):
        if base.startswith(prefix):
            base = base[len(prefix) :]
    for suffix in ("_serial", "_parallel", "_optimized_reductions", "_optimized"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def group_rows(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["shape"], row["dtype"])].append(row)
    return dict(groups)


def print_backend_comparison(rows: list[dict[str, str]]) -> None:
    by_backend: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    if len(by_backend) < 2:
        return
    print("  backend best:")
    for backend, backend_rows in sorted(by_backend.items()):
        best = min(backend_rows, key=latency)
        print(f"    - {backend}: {short(best)}")


def print_torch_tilelang_pairs(rows: list[dict[str, str]]) -> None:
    by_base: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_base[normalize_op_name(row["name"])].append(row)

    printed = False
    for base, candidates in sorted(by_base.items()):
        torch_rows = [row for row in candidates if row["backend"] == "torch"]
        tilelang_rows = [row for row in candidates if row["backend"] == "tilelang"]
        if not torch_rows or not tilelang_rows:
            continue
        torch_best = min(torch_rows, key=latency)
        tilelang_best = min(tilelang_rows, key=latency)
        speedup = latency(torch_best) / latency(tilelang_best)
        if not printed:
            print("  torch vs tilelang:")
            printed = True
        print(f"    - {base}: {speedup:.2f}x, {short(torch_best)} -> {short(tilelang_best)}")


def print_serial_parallel_pairs(rows: list[dict[str, str]]) -> None:
    by_base: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_base[normalize_op_name(row["name"])].append(row)

    printed = False
    for base, candidates in sorted(by_base.items()):
        serial = [row for row in candidates if row["name"].endswith("_serial")]
        parallel = [row for row in candidates if row["name"].endswith("_parallel")]
        if not serial or not parallel:
            continue
        serial_best = min(serial, key=latency)
        parallel_best = min(parallel, key=latency)
        speedup = latency(serial_best) / latency(parallel_best)
        if not printed:
            print("  serial vs parallel:")
            printed = True
        print(f"    - {base}: {speedup:.2f}x, {short(serial_best)} -> {short(parallel_best)}")
        if latency(parallel_best) > latency(serial_best):
            print(
                "      note: parallel is slower here; small shape, launch overhead, empty threads, "
                "or reduction synchronization can dominate."
            )


def print_optimized_pairs(rows: list[dict[str, str]]) -> None:
    baseline = [row for row in rows if row["name"] == "tilelang_decoder_block"]
    optimized = [row for row in rows if "optimized" in row["name"]]
    if not baseline or not optimized:
        return
    baseline_best = min(baseline, key=latency)
    optimized_best = min(optimized, key=latency)
    speedup = latency(baseline_best) / latency(optimized_best)
    print("  baseline vs optimized:")
    print(f"    - decoder_block: {speedup:.2f}x, {short(baseline_best)} -> {short(optimized_best)}")


def summarize(rows: list[dict[str, str]]) -> None:
    print(f"rows: {len(rows)}")
    for (suite, shape, dtype), group in sorted(group_rows(rows).items()):
        fastest = min(group, key=latency)
        slowest = max(group, key=latency)
        print(f"\n== suite={suite or 'unknown'} shape={shape or 'unknown'} dtype={dtype or 'unknown'} ==")
        print(f"  fastest: {short(fastest)}")
        print(f"  slowest: {short(slowest)}")
        print_backend_comparison(group)
        print_torch_tilelang_pairs(group)
        print_serial_parallel_pairs(group)
        print_optimized_pairs(group)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="benchmark CSV path")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        print("CSV has no benchmark rows")
        return 0
    print(f"csv: {args.csv}")
    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

