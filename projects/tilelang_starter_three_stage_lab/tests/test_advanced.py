from __future__ import annotations

import os

import pytest
import torch

from starter_tilelab.advanced import gemm_reference, gemm_tilelang


def test_gemm_reference() -> None:
    torch.manual_seed(0)
    a = torch.randn((16, 32))
    b = torch.randn((32, 16))
    torch.testing.assert_close(gemm_reference(a, b), a @ b)


def test_gemm_rejects_bad_rank_and_shape() -> None:
    with pytest.raises(ValueError, match="rank-2"):
        gemm_reference(torch.randn(2, 3, 4), torch.randn(4, 5))
    with pytest.raises(ValueError, match="shape"):
        gemm_reference(torch.randn(2, 3), torch.randn(4, 5))


def test_gemm_tilelang_rejects_non_aligned_shape_before_compile() -> None:
    a = torch.randn((17, 32), dtype=torch.float16)
    b = torch.randn((32, 16), dtype=torch.float16)
    with pytest.raises(ValueError, match="tile-aligned"):
        gemm_tilelang(a, b)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_gemm_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(1)
    a = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 16), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(gemm_tilelang(a, b), a @ b, rtol=2e-2, atol=2e-2)
