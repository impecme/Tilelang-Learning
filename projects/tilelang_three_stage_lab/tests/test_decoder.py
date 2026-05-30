from __future__ import annotations

from dataclasses import replace
import os

import pytest
import torch

from tilelab.decoder import (
    MiniDecoderConfig,
    decoder_block_reference,
    decoder_block_tilelang,
    decoder_block_tilelang_optimized,
    make_random_weights,
    mini_inference_reference,
    mini_inference_tilelang,
    mini_inference_tilelang_optimized,
)


def test_decoder_reference_smoke() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if torch.cuda.is_available() else "float32"
    config = MiniDecoderConfig(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        ffn_hidden_size=1024,
        vocab_size=4096,
        seq_len=128,
        dtype=dtype,
    )
    torch.manual_seed(0)
    weights = make_random_weights(config, seed=1, device=device)
    x = torch.randn((1, config.seq_len, config.hidden_size), device=device, dtype=getattr(torch, dtype))

    block_out = decoder_block_reference(x, weights, config)
    logits = mini_inference_reference(x, weights, config)

    assert block_out.shape == (1, 128, 256)
    assert logits.shape == (1, 128, 4096)


def test_decoder_rejects_bad_weight_shape() -> None:
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float32",
    )
    weights = make_random_weights(config, seed=4, device="cpu")
    bad_weights = replace(weights, qkv_weight=torch.randn((127, 384), dtype=torch.float32))
    x = torch.randn((1, 128, 128), dtype=torch.float32)

    with pytest.raises(ValueError, match="qkv_weight"):
        decoder_block_reference(x, bad_weights, config)


def test_decoder_rejects_bad_weight_dtype() -> None:
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float32",
    )
    weights = make_random_weights(config, seed=5, device="cpu")
    bad_weights = replace(weights, norm1_weight=weights.norm1_weight.half())
    x = torch.randn((1, 128, 128), dtype=torch.float32)

    with pytest.raises(ValueError, match="norm1_weight"):
        decoder_block_reference(x, bad_weights, config)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_decoder_rejects_bad_weight_device() -> None:
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float16",
    )
    weights = make_random_weights(config, seed=6, device="cpu")
    x = torch.randn((1, 128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="device"):
        decoder_block_reference(x, weights, config)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_decoder_tilelang_tiny_smoke() -> None:
    pytest.importorskip("tilelang")
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float16",
    )
    torch.manual_seed(2)
    weights = make_random_weights(config, seed=3, device="cuda")
    x = torch.randn((1, config.seq_len, config.hidden_size), device="cuda", dtype=torch.float16)

    actual = decoder_block_tilelang(x, weights, config)
    expected = decoder_block_reference(x, weights, config)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    optimized = decoder_block_tilelang_optimized(x, weights, config)
    torch.testing.assert_close(optimized, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.cuda
@pytest.mark.tilelang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.getenv("RUN_TILELANG_SMOKE") != "1", reason="set RUN_TILELANG_SMOKE=1")
def test_mini_inference_tilelang_tiny_logits_smoke() -> None:
    pytest.importorskip("tilelang")
    config = MiniDecoderConfig(
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        ffn_hidden_size=128,
        vocab_size=128,
        seq_len=128,
        dtype="float16",
    )
    torch.manual_seed(7)
    weights = make_random_weights(config, seed=8, device="cuda")
    x = torch.randn((1, config.seq_len, config.hidden_size), device="cuda", dtype=torch.float16)

    actual = mini_inference_tilelang(x, weights, config)
    optimized = mini_inference_tilelang_optimized(x, weights, config)
    expected = mini_inference_reference(x, weights, config)

    assert actual.shape == (1, 128, 128)
    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(optimized, expected, rtol=4e-2, atol=4e-2)
