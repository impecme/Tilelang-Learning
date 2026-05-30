from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_required_entry_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
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
        "starter_tilelab/basic.py",
        "starter_tilelab/advanced.py",
        "starter_tilelab/model.py",
        "scripts/check_starter.py",
        "scripts/run_starter_lab.py",
    ]
    missing = [path for path in expected if not (root / path).exists()]
    assert missing == []


def test_stage_readmes_are_small_learning_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    for relpath in [
        "01_first_kernel/README.md",
        "02_first_matrix_op/README.md",
        "03_first_model_flow/README.md",
    ]:
        text = (root / relpath).read_text(encoding="utf-8")
        assert "## 目标" in text
        assert "## 要读的代码" in text
        assert "## 运行命令" in text
        assert "## 必须回答的 3 个问题" in text
        assert "## 下一阶段入口" in text


def test_lab_readmes_are_actionable() -> None:
    root = Path(__file__).resolve().parents[1]
    for relpath in [
        "labs/01_vector_add_lab/README.md",
        "labs/02_gemm_lab/README.md",
        "labs/03_tiny_model_lab/README.md",
    ]:
        text = (root / relpath).read_text(encoding="utf-8")
        assert "## 目标" in text
        assert "## 要读的代码" in text
        assert "## 运行命令" in text
        assert "## 动手改法" in text
        assert "## 必须回答的 3 个问题" in text


def test_answer_key_exists_and_mentions_all_stages() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "labs/answer_key.md").read_text(encoding="utf-8")
    assert "Vector Add" in text
    assert "GEMM" in text
    assert "Tiny Model Flow" in text


def test_check_starter_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check_starter.py", "--stage", "1"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "01 First Kernel" in result.stdout
    assert "RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py" in result.stdout


def test_run_starter_lab_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    list_result = subprocess.run(
        [sys.executable, "scripts/run_starter_lab.py", "--list"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert list_result.returncode == 0, list_result.stderr
    assert "Vector Add" in list_result.stdout

    run_result = subprocess.run(
        [sys.executable, "scripts/run_starter_lab.py", "--lab", "all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "starter labs ok" in run_result.stdout
