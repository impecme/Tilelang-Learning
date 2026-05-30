from __future__ import annotations

import os

import pytest
import torch

from tilelab.advanced import (
    add_bias_reference,
    add_bias_tilelang,
    causal_mask_reference,
    causal_mask_tilelang,
    gelu_reference,
    gelu_tilelang,
    gemm_reference,
    gemm_tilelang,
    linear_bias_reference,
    linear_bias_gelu_reference,
    linear_bias_gelu_tilelang,
    linear_bias_tilelang,
    rmsnorm_parallel_reference,
    rmsnorm_parallel_tilelang,
    rmsnorm_reference,
    rmsnorm_tilelang,
    row_softmax_parallel_reference,
    row_softmax_parallel_tilelang,
    row_softmax_reference,
    row_softmax_tilelang,
    scale_causal_mask_reference,
    scale_causal_mask_tilelang,
    scale_reference,
    scale_tilelang,
)


def test_advanced_references() -> None:
    torch.manual_seed(0)
    a = torch.randn((8, 16))
    b = torch.randn((16, 4))
    torch.testing.assert_close(gemm_reference(a, b), a @ b)

    x = torch.randn((5, 9), dtype=torch.float16)
    torch.testing.assert_close(row_softmax_reference(x), torch.softmax(x.float(), dim=-1).half())
    torch.testing.assert_close(row_softmax_parallel_reference(x), row_softmax_reference(x))
    torch.testing.assert_close(gelu_reference(x), torch.nn.functional.gelu(x.float()).half())

    weight = torch.randn((9,), dtype=torch.float16)
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5) * weight.float()
    torch.testing.assert_close(rmsnorm_reference(x, weight), expected.half())
    torch.testing.assert_close(rmsnorm_parallel_reference(x, weight), rmsnorm_reference(x, weight))


def test_optimization_references() -> None:
    torch.manual_seed(10)
    x = torch.randn((4, 8), dtype=torch.float16)
    bias = torch.randn((8,), dtype=torch.float16)
    weight = torch.randn((8, 8), dtype=torch.float16)
    scores = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    torch.testing.assert_close(scale_reference(x, 0.25), x * 0.25)
    torch.testing.assert_close(add_bias_reference(x, bias), x + bias)
    torch.testing.assert_close(linear_bias_reference(x, weight, bias), x @ weight + bias)
    torch.testing.assert_close(
        scale_causal_mask_reference(scores, 0.5, seq_len=4),
        causal_mask_reference(scores * 0.5, seq_len=4),
    )


def test_causal_mask_reference() -> None:
    scores = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    actual = causal_mask_reference(scores, seq_len=4)
    assert actual[0, 1].item() == -65504.0
    assert actual[3, 3].item() == scores[3, 3].item()


def test_gemm_tilelang_rejects_non_aligned_shape_before_compile() -> None:
    a = torch.randn((96, 128), dtype=torch.float16)
    b = torch.randn((128, 128), dtype=torch.float16)

    with pytest.raises(ValueError, match="tile-aligned"):
        gemm_tilelang(a, b)


def test_parallel_reduction_shape_guards() -> None:
    x = torch.randn((4, 17), dtype=torch.float32)
    weight = torch.randn((17,), dtype=torch.float32)

    with pytest.raises(ValueError, match="power of two"):
        row_softmax_parallel_tilelang(x, block_n=24)

    with pytest.raises(ValueError, match="cols must be <= block_n"):
        row_softmax_parallel_tilelang(x, block_n=16)

    with pytest.raises(ValueError, match="block_n <= 1024"):
        rmsnorm_parallel_tilelang(x, weight, block_n=2048)

    with pytest.raises(ValueError, match="weight must have shape"):
        rmsnorm_parallel_tilelang(x, weight[:8], block_n=32)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_gemm_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(1)
    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(gemm_tilelang(a, b), a @ b, rtol=1e-2, atol=1e-2)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_reduction_and_activation_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(2)
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    w = torch.randn((16,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(row_softmax_tilelang(x), row_softmax_reference(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        row_softmax_parallel_tilelang(x, block_n=16),
        row_softmax_reference(x),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(rmsnorm_tilelang(x, w), rmsnorm_reference(x, w), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        rmsnorm_parallel_tilelang(x, w, block_n=16),
        rmsnorm_reference(x, w),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(gelu_tilelang(x), gelu_reference(x), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_linear_bias_gelu_and_mask_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(3)
    x = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    weight = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    bias = torch.randn((128,), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        linear_bias_gelu_tilelang(x, weight, bias),
        linear_bias_gelu_reference(x, weight, bias),
        rtol=2e-2,
        atol=2e-2,
    )

    scores = torch.randn((8, 8), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(causal_mask_tilelang(scores, 8), causal_mask_reference(scores, 8))


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_optimization_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    torch.manual_seed(11)
    x = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    weight = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    bias = torch.randn((128,), device="cuda", dtype=torch.float16)
    scores = torch.randn((8, 8), device="cuda", dtype=torch.float32)

    torch.testing.assert_close(scale_tilelang(x, 0.25), scale_reference(x, 0.25))
    torch.testing.assert_close(add_bias_tilelang(x, bias), add_bias_reference(x, bias))
    torch.testing.assert_close(
        linear_bias_tilelang(x, weight, bias),
        linear_bias_reference(x, weight, bias),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        scale_causal_mask_tilelang(scores, 0.5, 8),
        scale_causal_mask_reference(scores, 0.5, 8),
    )
