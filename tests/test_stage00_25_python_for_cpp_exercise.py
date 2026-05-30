from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from notes.exercise.stage00_25_python_for_cpp.task1 import ceildiv
from notes.exercise.stage00_25_python_for_cpp.task6 import row_sum_reference


def test_ceildiv_examples() -> None:
    assert ceildiv(1000, 256) == 4
    assert ceildiv(1024, 256) == 4
    assert ceildiv(0, 256) == 0


@pytest.mark.parametrize("a,b", [(-1, 256), (1, 0), (1, -2)])
def test_ceildiv_rejects_invalid_input(a: int, b: int) -> None:
    with pytest.raises(ValueError):
        ceildiv(a, b)


def test_row_sum_reference_matches_torch() -> None:
    x = torch.randn((3, 5), dtype=torch.float16)
    torch.testing.assert_close(row_sum_reference(x), x.float().sum(dim=-1))


def test_row_sum_reference_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        row_sum_reference(torch.randn((2, 3, 4)))


def test_matmul_shape() -> None:
    a = torch.randn((2, 3))
    b = torch.randn((3, 4))
    c = a @ b
    assert tuple(c.shape) == (2, 4)


def test_attention_scores_shape() -> None:
    q = torch.randn((2, 8, 16, 64))
    k = torch.randn((2, 8, 16, 64))
    scores = q @ k.transpose(-2, -1)
    assert tuple(scores.shape) == (2, 8, 16, 16)


def test_broadcast_shapes() -> None:
    x = torch.randn((2, 3))
    a = torch.randn(3)
    b = torch.randn((2, 1))
    assert tuple((x + a).shape) == (2, 3)
    assert tuple((x + b).shape) == (2, 3)

    with pytest.raises(RuntimeError):
        _ = x + torch.randn(2)


def test_stage00_25_runner_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    list_result = subprocess.run(
        [sys.executable, "scripts/run_stage00_25_exercise.py", "--list"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert list_result.returncode == 0, list_result.stderr
    assert "Stage 00.25 exercises" in list_result.stdout

    run_result = subprocess.run(
        [sys.executable, "scripts/run_stage00_25_exercise.py", "--task", "all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "Stage 00.25 exercises ok" in run_result.stdout
