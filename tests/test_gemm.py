from __future__ import annotations

import os

import pytest
import torch

from kernels.gemm import matmul_reference, matmul_tilelang


def test_matmul_reference() -> None:
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    torch.testing.assert_close(matmul_reference(a, b), a @ b)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_matmul_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(0)
    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    actual = matmul_tilelang(a, b, block_m=128, block_n=128, block_k=32)
    expected = a @ b

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
