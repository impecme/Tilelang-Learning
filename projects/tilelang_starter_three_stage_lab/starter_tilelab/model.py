"""Stage 03: a tiny model flow without attention."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from .advanced import gemm_tilelang
from .basic import vector_add_tilelang
from .common import dtype_name, load_tilelang, require_cuda_contiguous, torch_dtype


@dataclass(frozen=True)
class TinyModelConfig:
    seq_len: int = 16
    hidden_size: int = 64
    ffn_hidden_size: int = 128
    vocab_size: int = 256
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.seq_len <= 0 or self.hidden_size <= 0:
            raise ValueError("seq_len and hidden_size must be positive")
        if self.ffn_hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("ffn_hidden_size and vocab_size must be positive")
        torch_dtype(self.dtype)


@dataclass(frozen=True)
class TinyModelWeights:
    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    lm_head_weight: torch.Tensor


def _randn(shape: tuple[int, ...], dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * 0.02).to(dtype=dtype)


def make_tiny_weights(
    config: TinyModelConfig,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> TinyModelWeights:
    torch.manual_seed(seed)
    dtype = torch_dtype(config.dtype)
    h = config.hidden_size
    f = config.ffn_hidden_size
    v = config.vocab_size
    return TinyModelWeights(
        fc1_weight=_randn((h, f), dtype, device),
        fc2_weight=_randn((f, h), dtype, device),
        lm_head_weight=_randn((h, v), dtype, device),
    )


def _check_x(x: torch.Tensor, config: TinyModelConfig) -> None:
    expected = (config.seq_len, config.hidden_size)
    if tuple(x.shape) != expected:
        raise ValueError(f"x must have shape {expected}, got {tuple(x.shape)}")
    expected_dtype = torch_dtype(config.dtype)
    if x.dtype != expected_dtype:
        raise ValueError(f"x must have dtype {expected_dtype}, got {x.dtype}")


def _check_weights(weights: TinyModelWeights, config: TinyModelConfig, x: torch.Tensor) -> None:
    dtype = torch_dtype(config.dtype)
    device = x.device
    expected = {
        "fc1_weight": ((config.hidden_size, config.ffn_hidden_size), weights.fc1_weight),
        "fc2_weight": ((config.ffn_hidden_size, config.hidden_size), weights.fc2_weight),
        "lm_head_weight": ((config.hidden_size, config.vocab_size), weights.lm_head_weight),
    }
    for name, (shape, tensor) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


@lru_cache(maxsize=32)
def _compile_gelu(numel: int, block_size: int, dtype: str):
    tilelang, T = load_tilelang()

    @tilelang.jit
    def _factory(N: int, block: int = 256, dtype_name: str = "float32"):
        @T.prim_func
        def _kernel(X: T.Tensor((N,), dtype_name), Y: T.Tensor((N,), dtype_name)):
            with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
                for i in T.Parallel(block):
                    idx = bx * block + i
                    if idx < N:
                        value = X[idx]
                        Y[idx] = 0.5 * value * (1.0 + T.erf(value * 0.7071067811865476))

        return _kernel

    return _factory(numel, block_size, dtype)


def _gelu_tilelang(x: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    require_cuda_contiguous(x)
    out = torch.empty_like(x)
    kernel = _compile_gelu(x.numel(), block_size, dtype_name(x.dtype))
    kernel(x.reshape(-1), out.reshape(-1))
    return out


def tiny_model_reference(
    x: torch.Tensor,
    weights: TinyModelWeights,
    config: TinyModelConfig,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    hidden = torch.nn.functional.gelu((x @ weights.fc1_weight).float()).to(dtype=x.dtype)
    projected = hidden @ weights.fc2_weight
    residual = (x + projected).to(dtype=x.dtype)
    return residual @ weights.lm_head_weight


def tiny_model_tilelang(
    x: torch.Tensor,
    weights: TinyModelWeights,
    config: TinyModelConfig,
) -> torch.Tensor:
    _check_x(x, config)
    _check_weights(weights, config, x)
    require_cuda_contiguous(x, weights.fc1_weight, weights.fc2_weight, weights.lm_head_weight)
    hidden = _gelu_tilelang(gemm_tilelang(x, weights.fc1_weight))
    projected = gemm_tilelang(hidden, weights.fc2_weight)
    residual = vector_add_tilelang(x, projected)
    return gemm_tilelang(residual, weights.lm_head_weight)
