import math
import torch


def main() -> None:
    torch.manual_seed(0)

    q = torch.randn(4, 8)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)

    print("QK^T -> softmax -> P@V 的直觉解释")
    print("1. QK^T: 用每个 Query 去匹配所有 Key，得到 token-to-token 分数。")
    print("2. softmax: 把每一行分数变成权重，权重和为 1。")
    print("3. P@V: 用这些权重对 Value 做加权汇总，得到新的 token 表示。")

    scores = q @ k.T / math.sqrt(q.shape[-1])
    p = torch.softmax(scores, dim=-1)
    out = p @ v

    print("\nShape trace:")
    print("Q:", q.shape)
    print("K.T:", k.T.shape)
    print("scores = Q @ K.T:", scores.shape)
    print("P = softmax(scores):", p.shape)
    print("V:", v.shape)
    print("out = P @ V:", out.shape)

    print("\nOne row meaning:")
    row = 0
    print("P[0] 是第 0 个 token 对所有 token 的注意力权重:")
    print(p[row])
    print("P[0].sum():", p[row].sum())
    print("out[0] 是按 P[0] 加权汇总所有 V 后得到的新表示。")

    assert scores.shape == (4, 4)
    assert p.shape == (4, 4)
    assert out.shape == (4, 8)
    torch.testing.assert_close(p[row].sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    main()
