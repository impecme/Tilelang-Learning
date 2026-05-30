from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "01_first_kernel/README.md",
    "02_first_matrix_op/README.md",
    "03_first_model_flow/README.md",
    "labs/README.md",
    "labs/answer_key.md",
    "labs/01_vector_add_lab/README.md",
    "labs/02_gemm_lab/README.md",
    "labs/03_tiny_model_lab/README.md",
    "starter_tilelab/__init__.py",
    "starter_tilelab/basic.py",
    "starter_tilelab/advanced.py",
    "starter_tilelab/model.py",
    "tests/test_basic.py",
    "tests/test_advanced.py",
    "tests/test_model.py",
    "scripts/check_starter.py",
    "scripts/run_starter_lab.py",
]


STAGES = {
    "1": {
        "name": "01 First Kernel",
        "path": "01_first_kernel/README.md",
        "code": "starter_tilelab/basic.py",
        "test": "tests/test_basic.py",
        "lab": "labs/01_vector_add_lab/README.md",
        "command": "RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -q -m tilelang",
        "question": "能解释 idx 和 boundary guard",
    },
    "2": {
        "name": "02 First Matrix Op",
        "path": "02_first_matrix_op/README.md",
        "code": "starter_tilelab/advanced.py",
        "test": "tests/test_advanced.py",
        "lab": "labs/02_gemm_lab/README.md",
        "command": "RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -q -m tilelang",
        "question": "能解释 M/N/K 和 tile-aligned shape",
    },
    "3": {
        "name": "03 First Model Flow",
        "path": "03_first_model_flow/README.md",
        "code": "starter_tilelab/model.py",
        "test": "tests/test_model.py",
        "lab": "labs/03_tiny_model_lab/README.md",
        "command": "RUN_TILELANG_SMOKE=1 pytest tests/test_model.py -q -m tilelang",
        "question": "能从 x 推导到 logits",
    },
}


def print_environment() -> None:
    print("== Environment ==")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        print(f"torch cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"gpu[{index}]: {torch.cuda.get_device_name(index)}")
    except Exception as exc:
        print(f"torch import failed: {exc}")

    try:
        print(f"tilelang: {importlib.metadata.version('tilelang')}")
    except importlib.metadata.PackageNotFoundError:
        print("tilelang: not installed")


def print_required_files() -> bool:
    print("\n== Required Files ==")
    ok = True
    for relpath in REQUIRED_PATHS:
        exists = (ROOT / relpath).exists()
        print(f"{'ok  ' if exists else 'MISS'} {relpath}")
        ok = ok and exists
    return ok


def print_stage(stage_id: str) -> bool:
    selected = STAGES.items() if stage_id == "all" else [(stage_id, STAGES[stage_id])]
    ok = True
    print("\n== Stage Checks ==")
    for sid, stage in selected:
        print(f"\n[{sid}] {stage['name']}")
        for key in ("path", "code", "test", "lab"):
            relpath = stage[key]
            exists = (ROOT / relpath).exists()
            print(f"{'ok  ' if exists else 'MISS'} {relpath}")
            ok = ok and exists
        print(f"command: {stage['command']}")
        print(f"gate: {stage['question']}")
    return ok


def print_commands() -> None:
    print("\n== Recommended Commands ==")
    print("pytest -q")
    print("RUN_TILELANG_SMOKE=1 pytest -q -m tilelang")
    print("python3 scripts/run_starter_lab.py --list")
    print("python3 scripts/run_starter_lab.py --lab all")
    print("open labs/answer_key.md after each lab to compare expected observations")
    print("python3 scripts/check_starter.py --stage 1")
    print("python3 scripts/check_starter.py --stage 2")
    print("python3 scripts/check_starter.py --stage 3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
    args = parser.parse_args()

    print(f"project: {ROOT}")
    print_environment()
    files_ok = print_required_files()
    stage_ok = print_stage(args.stage)
    print_commands()
    return 0 if files_ok and stage_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
