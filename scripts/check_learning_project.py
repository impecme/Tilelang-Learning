from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "notes/learning_map.md",
    "notes/stage00_25_python_for_cpp.md",
    "notes/exercise/stage00_25_python_for_cpp/task1.py",
    "notes/exercise/stage00_25_python_for_cpp/task2.py",
    "notes/exercise/stage00_25_python_for_cpp/task3.py",
    "notes/exercise/stage00_25_python_for_cpp/task4.py",
    "notes/exercise/stage00_25_python_for_cpp/task5.py",
    "notes/exercise/stage00_25_python_for_cpp/task6.py",
    "notes/exercise/stage00_25_python_for_cpp/task7.py",
    "notes/exercise/stage00_25_python_for_cpp/task8.py",
    "notes/exercise/stage00_25_python_for_cpp/task9.py",
    "scripts/check_env.py",
    "scripts/run_stage00_25_exercise.py",
    "tests/test_stage00_25_python_for_cpp_exercise.py",
]

PROJECT_REQUIRED_PATHS = [
    "projects/tilelang_starter_three_stage_lab/README.md",
    "projects/tilelang_starter_three_stage_lab/labs/README.md",
    "projects/tilelang_starter_three_stage_lab/labs/answer_key.md",
    "projects/tilelang_starter_three_stage_lab/scripts/check_starter.py",
    "projects/tilelang_starter_three_stage_lab/scripts/run_starter_lab.py",
    "projects/tilelang_three_stage_lab/README.md",
    "projects/tilelang_three_stage_lab/labs/README.md",
    "projects/tilelang_three_stage_lab/docs/lab_answer_key.md",
    "projects/tilelang_three_stage_lab/scripts/check_project.py",
    "projects/tilelang_three_stage_lab/scripts/run_lab.py",
    "projects/tilelang_three_stage_lab/scripts/summarize_bench_csv.py",
]


def check_paths(paths: list[str], title: str) -> bool:
    print(f"\n== {title} ==")
    ok = True
    for relpath in paths:
        exists = (ROOT / relpath).exists()
        print(f"{'ok  ' if exists else 'MISS'} {relpath}")
        ok = ok and exists
    return ok


def run_import_smoke() -> bool:
    print("\n== Import Smoke ==")
    try:
        from notes.exercise.stage00_25_python_for_cpp.task1 import ceildiv

        assert ceildiv(1000, 256) == 4
    except Exception as exc:
        print(f"MISS Stage 00.25 task import failed: {exc}")
        return False
    print("ok   Stage 00.25 task import")
    return True


def print_notes() -> None:
    print("\n== Notes ==")
    if not (ROOT / "notes/highlights/pythonbase.md").exists():
        print("info notes/highlights/pythonbase.md is absent; confirm this deletion before commit if needed.")


def print_commands(skip_projects: bool) -> None:
    print("\n== Recommended Commands ==")
    print("python3 scripts/run_stage00_25_exercise.py --list")
    print("python3 scripts/run_stage00_25_exercise.py --task all")
    print("pytest tests/test_stage00_25_python_for_cpp_exercise.py -q")
    print("pytest -q")
    if not skip_projects:
        print("cd projects/tilelang_starter_three_stage_lab && python3 scripts/check_starter.py")
        print("cd projects/tilelang_three_stage_lab && python3 scripts/check_project.py")


def run_command(command: list[str], cwd: Path) -> int:
    print(f"\n== Running: {' '.join(command)} ==")
    return subprocess.call(command, cwd=cwd)


def run_pytest(skip_projects: bool) -> int:
    commands = [([sys.executable, "-m", "pytest", "-q"], ROOT)]
    if not skip_projects:
        commands.extend(
            [
                ([sys.executable, "-m", "pytest", "-q"], ROOT / "projects/tilelang_starter_three_stage_lab"),
                ([sys.executable, "-m", "pytest", "-q"], ROOT / "projects/tilelang_three_stage_lab"),
            ]
        )
    for command, cwd in commands:
        code = run_command(command, cwd)
        if code != 0:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-projects", action="store_true", help="only check root notes and exercises")
    parser.add_argument("--run-pytest", action="store_true", help="also run pytest")
    args = parser.parse_args()

    print(f"project: {ROOT}")
    ok = check_paths(ROOT_REQUIRED_PATHS, "Root Learning Files")
    if not args.skip_projects:
        ok = check_paths(PROJECT_REQUIRED_PATHS, "Project Learning Files") and ok
    ok = run_import_smoke() and ok
    print_notes()
    print_commands(args.skip_projects)

    if args.run_pytest:
        pytest_code = run_pytest(args.skip_projects)
        if pytest_code != 0:
            return pytest_code
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
