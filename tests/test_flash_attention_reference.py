from __future__ import annotations

import pytest
import torch

from kernels.flash_attention import flash_attention_forward
from kernels.reference import naive_attention_forward, online_attention_forward, sdpa_attention_forward


pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 128, 64), (1, 2, 129, 64), (2, 4, 256, 128)])
def test_flash_attention_api_matches_naive(dtype: torch.dtype, shape: tuple[int, int, int, int]) -> None:
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    actual = flash_attention_forward(q, k, v)
    expected = naive_attention_forward(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_online_attention_reference_supports_causal() -> None:
    torch.manual_seed(1)
    q = torch.randn((1, 2, 127, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = online_attention_forward(q, k, v, causal=True, block_n=31)
    expected = naive_attention_forward(q, k, v, causal=True)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sdpa_baseline_is_close_to_naive() -> None:
    torch.manual_seed(2)
    q = torch.randn((1, 3, 128, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = sdpa_attention_forward(q, k, v)
    expected = naive_attention_forward(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

