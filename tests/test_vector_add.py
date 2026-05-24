from __future__ import annotations

import os

import pytest
import torch

from kernels.vector_add import vector_add_reference, vector_add_tilelang


def test_vector_add_reference() -> None:
    a = torch.randn(32)
    b = torch.randn(32)
    torch.testing.assert_close(vector_add_reference(a, b), a + b)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_vector_add_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(0)
    a = torch.randn(1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, device="cuda", dtype=torch.float32)

    actual = vector_add_tilelang(a, b)
    expected = a + b

    torch.testing.assert_close(actual, expected)

