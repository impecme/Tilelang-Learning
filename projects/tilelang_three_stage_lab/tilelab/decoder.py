"""Stage 03 mini decoder block inference pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .advanced import (
    gemm_reference,
    gemm_tilelang,
    linear_bias_tilelang,
    linear_bias_gelu_tilelang,
    rmsnorm_parallel_tilelang,
    rmsnorm_reference,
    rmsnorm_tilelang,
    row_softmax_parallel_tilelang,
    row_softmax_tilelang,
    scale_causal_mask_tilelang,
    scale_tilelang,
)
from .basic import vector_add_tilelang
from .common import require_cuda_contiguous


@dataclass(frozen=True)
class MiniDecoderConfig:
    hidden_size: int = 256
    num_heads: int = 4
    head_dim: int = 64
    ffn_hidden_size: int = 1024
    vocab_size: int = 4096
    seq_len: int = 128
    dtype: str = "float16"
    causal: bool = True
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if self.seq_len <= 0 or self.hidden_size <= 0 or self.ffn_hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("seq_len, hidden_size, ffn_hidden_size, and vocab_size must be positive")
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        _torch_dtype(self.dtype)


@dataclass(frozen=True)
class MiniDecoderWeights:
    norm1_weight: torch.Tensor
    qkv_weight: torch.Tensor
    qkv_bias: torch.Tensor
    out_weight: torch.Tensor
    out_bias: torch.Tensor
    norm2_weight: torch.Tensor
    mlp_up_weight: torch.Tensor
    mlp_up_bias: torch.Tensor
    mlp_down_weight: torch.Tensor
    mlp_down_bias: torch.Tensor
    lm_head_weight: torch.Tensor
    lm_head_bias: torch.Tensor


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype name: {name}")
    return mapping[name]


def _randn(shape: tuple[int, ...], dtype: torch.dtype, device: str | torch.device, scale: float) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(dtype=dtype)


def make_random_weights(
    config: MiniDecoderConfig,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> MiniDecoderWeights:
    torch.manual_seed(seed)
    dtype = _torch_dtype(config.dtype)
    h = config.hidden_size
    f = config.ffn_hidden_size
    v = config.vocab_size
    scale = 0.02
    return MiniDecoderWeights(
        norm1_weight=torch.ones((h,), device=device, dtype=dtype),
        qkv_weight=_randn((h, 3 * h), dtype, device, scale),
        qkv_bias=torch.zeros((3 * h,), device=device, dtype=dtype),
        out_weight=_randn((h, h), dtype, device, scale),
        out_bias=torch.zeros((h,), device=device, dtype=dtype),
        norm2_weight=torch.ones((h,), device=device, dtype=dtype),
        mlp_up_weight=_randn((h, f), dtype, device, scale),
        mlp_up_bias=torch.zeros((f,), device=device, dtype=dtype),
        mlp_down_weight=_randn((f, h), dtype, device, scale),
        mlp_down_bias=torch.zeros((h,), device=device, dtype=dtype),
        lm_head_weight=_randn((h, v), dtype, device, scale),
        lm_head_bias=torch.zeros((v,), device=device, dtype=dtype),
    )


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_weights(weights: MiniDecoderWeights, config: MiniDecoderConfig, x: torch.Tensor) -> None:
    dtype = _torch_dtype(config.dtype)
    device = x.device
    h = config.hidden_size
    f = config.ffn_hidden_size
    v = config.vocab_size
    expected = {
        "norm1_weight": ((h,), weights.norm1_weight),
        "qkv_weight": ((h, 3 * h), weights.qkv_weight),
        "qkv_bias": ((3 * h,), weights.qkv_bias),
        "out_weight": ((h, h), weights.out_weight),
        "out_bias": ((h,), weights.out_bias),
        "norm2_weight": ((h,), weights.norm2_weight),
        "mlp_up_weight": ((h, f), weights.mlp_up_weight),
        "mlp_up_bias": ((f,), weights.mlp_up_bias),
        "mlp_down_weight": ((f, h), weights.mlp_down_weight),
        "mlp_down_bias": ((h,), weights.mlp_down_bias),
        "lm_head_weight": ((h, v), weights.lm_head_weight),
        "lm_head_bias": ((v,), weights.lm_head_bias),
    }
    for name, (shape, tensor) in expected.items():
        _check_tensor(name, tensor, shape, dtype, device)


def _check_x(x: torch.Tensor, config: MiniDecoderConfig) -> None:
    if x.ndim != 3:
        raise ValueError("x must have shape (B, S, hidden_size)")
    if x.shape[1] != config.seq_len or x.shape[2] != config.hidden_size:
        raise ValueError("x shape does not match config")
    expected_dtype = _torch_dtype(config.dtype)
    if x.dtype != expected_dtype:
        raise ValueError(f"x must have dtype {expected_dtype}, got {x.dtype}")


def _linear_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weight + bias


def _split_qkv(qkv: torch.Tensor, config: MiniDecoderConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq, _ = qkv.shape
    qkv_5d = qkv.view(batch, seq, 3, config.num_heads, config.head_dim)
    q = qkv_5d[:, :, 0, :, :].permute(0, 2, 1, 3).contiguous()
    k = qkv_5d[:, :, 1, :, :].permute(0, 2, 1, 3).contiguous()
    v = qkv_5d[:, :, 2, :, :].permute(0, 2, 1, 3).contiguous()
    return q, k, v


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(config.head_dim)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if config.causal:
        row = torch.arange(config.seq_len, device=q.device).view(config.seq_len, 1)
        col = torch.arange(config.seq_len, device=q.device).view(1, config.seq_len)
        scores = scores.masked_fill(col > row, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float()).to(dtype=q.dtype)
    return out.permute(0, 2, 1, 3).contiguous().view(q.shape[0], config.seq_len, config.hidden_size)


def decoder_block_reference(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    batch, seq, hidden = x.shape
    x2d = x.reshape(batch * seq, hidden)
    norm1 = rmsnorm_reference(x2d, weights.norm1_weight, config.eps)
    qkv = _linear_reference(norm1, weights.qkv_weight, weights.qkv_bias).view(batch, seq, 3 * hidden)
    q, k, v = _split_qkv(qkv, config)
    attn = _attention_reference(q, k, v, config).reshape(batch * seq, hidden)
    out_proj = _linear_reference(attn, weights.out_weight, weights.out_bias)
    residual1 = (x2d + out_proj).to(dtype=x.dtype)
    norm2 = rmsnorm_reference(residual1, weights.norm2_weight, config.eps)
    mlp_hidden = torch.nn.functional.gelu(
        _linear_reference(norm2, weights.mlp_up_weight, weights.mlp_up_bias).float()
    ).to(dtype=x.dtype)
    mlp_out = _linear_reference(mlp_hidden, weights.mlp_down_weight, weights.mlp_down_bias)
    return (residual1 + mlp_out).to(dtype=x.dtype).view(batch, seq, hidden)


def _linear_tilelang(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return linear_bias_tilelang(x, weight, bias)


def _attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: MiniDecoderConfig,
    softmax_fn=row_softmax_tilelang,
) -> torch.Tensor:
    batch, heads, seq, dim = q.shape
    pieces: list[torch.Tensor] = []
    scale = 1.0 / math.sqrt(dim)
    for b in range(batch):
        head_pieces: list[torch.Tensor] = []
        for h in range(heads):
            q_mat = q[b, h].contiguous()
            k_mat = k[b, h].transpose(0, 1).contiguous()
            v_mat = v[b, h].contiguous()
            raw_scores = gemm_tilelang(q_mat, k_mat)
            if config.causal:
                scores = scale_causal_mask_tilelang(raw_scores, scale, seq)
            else:
                scores = scale_tilelang(raw_scores, scale)
            probs = softmax_fn(scores)
            head_pieces.append(gemm_tilelang(probs, v_mat, block_n=dim).view(1, seq, dim))
        pieces.append(torch.cat(head_pieces, dim=0).view(1, heads, seq, dim))
    out = torch.cat(pieces, dim=0)
    return out.permute(0, 2, 1, 3).contiguous().view(batch, seq, config.hidden_size)


def _decoder_block_tilelang_impl(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
    rmsnorm_fn,
    softmax_fn,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    require_cuda_contiguous(x)
    batch, seq, hidden = x.shape
    x2d = x.reshape(batch * seq, hidden).contiguous()

    norm1 = rmsnorm_fn(x2d, weights.norm1_weight, config.eps)
    qkv = _linear_tilelang(norm1, weights.qkv_weight, weights.qkv_bias).view(batch, seq, 3 * hidden)
    q, k, v = _split_qkv(qkv, config)
    attn = _attention_tilelang(q, k, v, config, softmax_fn=softmax_fn).reshape(batch * seq, hidden)
    out_proj = _linear_tilelang(attn, weights.out_weight, weights.out_bias)
    residual1 = vector_add_tilelang(x2d, out_proj)

    norm2 = rmsnorm_fn(residual1, weights.norm2_weight, config.eps)
    mlp_hidden = linear_bias_gelu_tilelang(norm2, weights.mlp_up_weight, weights.mlp_up_bias)
    mlp_down = _linear_tilelang(mlp_hidden, weights.mlp_down_weight, weights.mlp_down_bias)
    return vector_add_tilelang(residual1, mlp_down).view(batch, seq, hidden)


def decoder_block_tilelang(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    return _decoder_block_tilelang_impl(
        x,
        weights,
        config,
        rmsnorm_fn=rmsnorm_tilelang,
        softmax_fn=row_softmax_tilelang,
    )


def decoder_block_tilelang_optimized(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    return _decoder_block_tilelang_impl(
        x,
        weights,
        config,
        rmsnorm_fn=rmsnorm_parallel_tilelang,
        softmax_fn=row_softmax_parallel_tilelang,
    )


def mini_inference_reference(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    block_out = decoder_block_reference(x, weights, config).reshape(-1, config.hidden_size)
    logits = gemm_reference(block_out, weights.lm_head_weight) + weights.lm_head_bias
    return logits.view(x.shape[0], x.shape[1], config.vocab_size)


def mini_inference_tilelang(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    block_out = decoder_block_tilelang(x, weights, config).reshape(-1, config.hidden_size).contiguous()
    logits = _linear_tilelang(block_out, weights.lm_head_weight, weights.lm_head_bias)
    return logits.view(x.shape[0], x.shape[1], config.vocab_size)


def mini_inference_tilelang_optimized(
    x: torch.Tensor,
    weights: MiniDecoderWeights,
    config: MiniDecoderConfig,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    block_out = decoder_block_tilelang_optimized(x, weights, config).reshape(-1, config.hidden_size).contiguous()
    logits = _linear_tilelang(block_out, weights.lm_head_weight, weights.lm_head_bias)
    return logits.view(x.shape[0], x.shape[1], config.vocab_size)
