import math
import torch


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A small PyTorch attention reference for shape (S, D)."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q/k/v must be rank-2 tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have the same shape")

    d = q.shape[-1]
    scores = q @ k.T / math.sqrt(d)
    p = torch.softmax(scores, dim=-1)
    return p @ v


def main() -> None:
    torch.manual_seed(0)

    s = 4
    d = 8
    q = torch.randn(s, d)
    k = torch.randn(s, d)
    v = torch.randn(s, d)

    scores = q @ k.T / math.sqrt(d)
    p = torch.softmax(scores, dim=-1)
    out = p @ v
    ref = attention_reference(q, k, v)

    print("Q:", q.shape)
    print("K:", k.shape)
    print("V:", v.shape)
    print("scores:", scores.shape)
    print("P:", p.shape)
    print("out:", out.shape)
    print("P row sums:", p.sum(dim=-1))

    # softmax 后，每一行都是一个概率分布，行和应该接近 1。
    torch.testing.assert_close(p.sum(dim=-1), torch.ones(s), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    main()
