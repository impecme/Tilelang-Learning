from __future__ import annotations

import os

import pytest
import torch

from starter_tilelab.model import (
    TinyModelConfig,
    make_tiny_weights,
    tiny_model_reference,
    tiny_model_tilelang,
)


def test_tiny_model_reference_shape() -> None:
    config = TinyModelConfig(dtype="float32")
    torch.manual_seed(0)
    x = torch.randn((config.seq_len, config.hidden_size), dtype=torch.float32)
    weights = make_tiny_weights(config, seed=1, device="cpu")
    logits = tiny_model_reference(x, weights, config)
    assert tuple(logits.shape) == (config.seq_len, config.vocab_size)


def test_tiny_model_rejects_bad_input_shape() -> None:
    config = TinyModelConfig(dtype="float32")
    weights = make_tiny_weights(config, seed=1, device="cpu")
    with pytest.raises(ValueError, match="x must have shape"):
        tiny_model_reference(torch.randn((config.seq_len + 1, config.hidden_size)), weights, config)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_tiny_model_tilelang_smoke() -> None:
    pytest.importorskip("tilelang")
    config = TinyModelConfig()
    torch.manual_seed(2)
    x = torch.randn((config.seq_len, config.hidden_size), device="cuda", dtype=torch.float16)
    weights = make_tiny_weights(config, seed=3, device="cuda")
    torch.testing.assert_close(
        tiny_model_tilelang(x, weights, config),
        tiny_model_reference(x, weights, config),
        rtol=5e-2,
        atol=5e-2,
    )
