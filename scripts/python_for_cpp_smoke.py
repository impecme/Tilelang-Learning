from __future__ import annotations

import math
from functools import wraps

import torch


def banner(name: str) -> None:
    print(f"\n== {name} ==")


def demo_python_reference_semantics() -> None:
    banner("Python names bind to objects")
    a = [1, 2, 3]
    b = a
    b.append(4)
    print("a after b.append(4):", a)

    c = a.copy()
    c.append(5)
    print("a after c.append(5):", a)
    print("c:", c)


def trace_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"calling {fn.__name__}")
        return fn(*args, **kwargs)

    return wrapper


@trace_call
def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def demo_decorator_and_function() -> None:
    banner("Decorator and ceildiv")
    print("ceildiv(1000, 256):", ceildiv(1000, 256))


def demo_tensor_basics() -> None:
    banner("PyTorch tensor basics")
    x = torch.randn((2, 3, 4))
    print("x.shape:", tuple(x.shape))
    print("x.dtype:", x.dtype)
    print("x.device:", x.device)
    print("x.ndim:", x.ndim)
    print("x.is_contiguous:", x.is_contiguous())

    xt = x.transpose(-2, -1)
    print("xt.shape after transpose(-2, -1):", tuple(xt.shape))
    print("xt.is_contiguous:", xt.is_contiguous())
    print("xt.contiguous().is_contiguous:", xt.contiguous().is_contiguous())


def demo_matmul_and_attention_shapes() -> None:
    banner("Matmul and attention shapes")
    batch, heads, seq, dim = 2, 8, 16, 64
    q = torch.randn((batch, heads, seq, dim))
    k = torch.randn((batch, heads, seq, dim))
    v = torch.randn((batch, heads, seq, dim))

    scores = q @ k.transpose(-2, -1) / math.sqrt(dim)
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v

    print("q:", tuple(q.shape))
    print("k.transpose(-2, -1):", tuple(k.transpose(-2, -1).shape))
    print("scores = q @ k^T:", tuple(scores.shape))
    print("probs:", tuple(probs.shape))
    print("out = probs @ v:", tuple(out.shape))
    print("max row-sum error:", (probs.sum(dim=-1) - 1).abs().max().item())

    assert scores.shape == (batch, heads, seq, seq)
    assert out.shape == (batch, heads, seq, dim)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones((batch, heads, seq)))


def main() -> None:
    demo_python_reference_semantics()
    demo_decorator_and_function()
    demo_tensor_basics()
    demo_matmul_and_attention_shapes()


if __name__ == "__main__":
    main()

