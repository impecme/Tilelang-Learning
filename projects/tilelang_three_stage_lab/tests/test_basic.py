from __future__ import annotations

import os

import pytest
import torch

from tilelab.basic import (
    axpy_reference,
    axpy_tilelang,
    copy_reference,
    copy_tilelang,
    row_sum_parallel_reference,
    row_sum_parallel_tilelang,
    row_sum_reference,
    row_sum_tilelang,
    vector_add_reference,
    vector_add_tilelang,
)


@pytest.mark.parametrize("numel", [1024, 1000])
def test_elementwise_references(numel: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(numel)
    b = torch.randn(numel)
    torch.testing.assert_close(vector_add_reference(a, b), a + b)
    torch.testing.assert_close(copy_reference(a), a)
    torch.testing.assert_close(axpy_reference(2.5, a, b), 2.5 * a + b)


def test_row_sum_reference() -> None:
    x = torch.randn((7, 19), dtype=torch.float16)
    torch.testing.assert_close(row_sum_reference(x), x.float().sum(dim=-1))
    torch.testing.assert_close(row_sum_parallel_reference(x), row_sum_reference(x))


def test_row_sum_parallel_shape_guards() -> None:
    x = torch.randn((4, 17), dtype=torch.float32)

    with pytest.raises(ValueError, match="power of two"):
        row_sum_parallel_tilelang(x, block_n=24)

    with pytest.raises(ValueError, match="cols must be <= block_n"):
        row_sum_parallel_tilelang(x, block_n=16)

    with pytest.raises(ValueError, match="block_n <= 1024"):
        row_sum_parallel_tilelang(x, block_n=2048)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
@pytest.mark.parametrize("numel", [1024, 1000])
def test_basic_tilelang_elementwise(numel: int) -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(1)
    a = torch.randn(numel, device="cuda", dtype=torch.float32)
    b = torch.randn(numel, device="cuda", dtype=torch.float32)

    torch.testing.assert_close(vector_add_tilelang(a, b), a + b)
    torch.testing.assert_close(copy_tilelang(a), a)
    torch.testing.assert_close(axpy_tilelang(1.25, a, b), 1.25 * a + b)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_row_sum_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(2)
    x = torch.randn((8, 17), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(row_sum_tilelang(x), row_sum_reference(x))
    torch.testing.assert_close(row_sum_parallel_tilelang(x, block_n=32), row_sum_reference(x))
