from __future__ import annotations

import os

import pytest
import torch

from starter_tilelab.basic import vector_add_reference, vector_add_tilelang


@pytest.mark.parametrize("numel", [1024, 1000])
def test_vector_add_reference(numel: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(numel)
    b = torch.randn(numel)
    torch.testing.assert_close(vector_add_reference(a, b), a + b)


def test_vector_add_reference_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        vector_add_reference(torch.randn(4), torch.randn(5))


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
@pytest.mark.parametrize("numel", [1024, 1000])
def test_vector_add_tilelang_smoke(numel: int) -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(1)
    a = torch.randn(numel, device="cuda", dtype=torch.float32)
    b = torch.randn(numel, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(vector_add_tilelang(a, b), a + b)
