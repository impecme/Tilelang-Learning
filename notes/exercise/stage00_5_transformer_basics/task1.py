import torch


def main() -> None:
    torch.manual_seed(0)

    # Shape 练习：
    # Q/K/V 都是二维矩阵，shape = (S, D)。
    # S = 128 表示 token 数量，D = 64 表示每个 token 的向量维度。
    q = torch.randn(128, 64)
    k = torch.randn(128, 64)
    v = torch.randn(128, 64)

    # K.T 是二维矩阵转置：(128, 64) -> (64, 128)。
    scores = q @ k.T

    # softmax 不改变 shape，只把每一行分数变成概率分布。
    p = torch.softmax(scores, dim=-1)

    # P @ V: (128, 128) @ (128, 64) = (128, 64)。
    out = p @ v

    print("Q.shape:", q.shape)
    print("K.shape:", k.shape)
    print("V.shape:", v.shape)
    print("K.T.shape:", k.T.shape)
    print("scores = Q @ K.T shape:", scores.shape)
    print("P = softmax(scores) shape:", p.shape)
    print("out = P @ V shape:", out.shape)

    assert scores.shape == (128, 128)
    assert p.shape == (128, 128)
    assert out.shape == (128, 64)


if __name__ == "__main__":
    main()
