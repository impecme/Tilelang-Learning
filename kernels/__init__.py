"""Learning kernels and references for the TileLang study project."""

from importlib import import_module

_EXPORTS = {
    "flash_attention_forward": ("kernels.flash_attention", "flash_attention_forward"),
    "matmul_reference": ("kernels.gemm", "matmul_reference"),
    "matmul_tilelang": ("kernels.gemm", "matmul_tilelang"),
    "naive_attention_forward": ("kernels.reference", "naive_attention_forward"),
    "online_attention_forward": ("kernels.reference", "online_attention_forward"),
    "sdpa_attention_forward": ("kernels.reference", "sdpa_attention_forward"),
    "vector_add_reference": ("kernels.vector_add", "vector_add_reference"),
    "vector_add_tilelang": ("kernels.vector_add", "vector_add_tilelang"),
}

__all__ = [
    "flash_attention_forward",
    "matmul_reference",
    "matmul_tilelang",
    "naive_attention_forward",
    "online_attention_forward",
    "sdpa_attention_forward",
    "vector_add_reference",
    "vector_add_tilelang",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
