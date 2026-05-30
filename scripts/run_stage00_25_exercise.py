from __future__ import annotations

import argparse
import inspect
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXERCISE_DIR = ROOT / "notes" / "exercise" / "stage00_25_python_for_cpp"

TASKS = {
    "1": ("ceildiv and GPU block count", "task1.py"),
    "2": ("tensor metadata and contiguous", "task2.py"),
    "3": ("matmul basics", "task3.py"),
    "4": ("attention shapes", "task4.py"),
    "5": ("broadcasting rules", "task5.py"),
    "6": ("row_sum_reference", "task6.py"),
    "7": ("pytest thinking", "task7.py"),
    "8": ("entry guard", "task8.py"),
    "9": ("python_for_cpp smoke script", "task9.py"),
}


def list_tasks() -> None:
    print("Stage 00.25 exercises:")
    for task_id, (name, filename) in TASKS.items():
        print(f"  {task_id}: {name} ({filename})")


def load_task(task_id: str):
    name, filename = TASKS[task_id]
    path = EXERCISE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"stage00_25_task_{task_id}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    inserted: list[str] = []
    for item in reversed([str(ROOT), str(EXERCISE_DIR)]):
        if item not in sys.path:
            sys.path.insert(0, item)
            inserted.append(item)
    try:
        spec.loader.exec_module(module)
    finally:
        for item in inserted:
            if item in sys.path:
                sys.path.remove(item)
    return name, module


def run_task(task_id: str, use_cuda: bool) -> None:
    name, module = load_task(task_id)
    print(f"\n== Task {task_id}: {name} ==", flush=True)
    signature = inspect.signature(module.main)
    try:
        if "use_cuda" in signature.parameters:
            module.main(use_cuda=use_cuda)
        else:
            module.main()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list available tasks")
    parser.add_argument("--task", choices=("all",) + tuple(TASKS), default="all")
    parser.add_argument("--cuda", action="store_true", help="enable optional CUDA observations")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return 0

    if args.cuda:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available; rerun without --cuda for CPU-only exercises.")

    selected = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    for task_id in selected:
        run_task(task_id, args.cuda)
    print("\nStage 00.25 exercises ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
