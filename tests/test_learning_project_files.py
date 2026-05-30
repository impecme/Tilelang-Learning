from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_learning_map_and_stage00_25_entries_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
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
        "scripts/run_stage00_25_exercise.py",
        "scripts/check_learning_project.py",
    ]
    missing = [path for path in expected if not (root / path).exists()]
    assert missing == []


def test_root_docs_point_to_exercises() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    stage_doc = (root / "notes/stage00_25_python_for_cpp.md").read_text(encoding="utf-8")
    learning_map = (root / "notes/learning_map.md").read_text(encoding="utf-8")

    assert "scripts/run_stage00_25_exercise.py --task all" in readme
    assert "## 19. 小练习" in stage_doc
    assert "row_sum_reference" in stage_doc
    assert "tilelang_starter_three_stage_lab" in learning_map
    assert "tilelang_three_stage_lab" in learning_map


def test_root_learning_check_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check_learning_project.py", "--skip-projects"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Root Learning Files" in result.stdout
    assert "Stage 00.25 task import" in result.stdout
