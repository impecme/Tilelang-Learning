import torch


def main() -> None:
    torch.manual_seed(0)

    # Batch/head 练习：
    # Q.shape = (B, H, S, D)
    # B = 2, H = 8, S = 512, D = 64。
    q = torch.randn(2, 8, 512, 64)
    k = torch.randn(2, 8, 512, 64)
    v = torch.randn(2, 8, 512, 64)

    # k.transpose(-2, -1): (B,H,S,D) -> (B,H,D,S)
    scores = q @ k.transpose(-2, -1)
    p = torch.softmax(scores, dim=-1)
    out = p @ v

    print("Q.shape:", q.shape)
    print("K.transpose(-2, -1).shape:", k.transpose(-2, -1).shape)
    print("scores.shape:", scores.shape)
    print("P.shape:", p.shape)
    print("out.shape:", out.shape)

    # 最后两个维度做矩阵乘：
    # (512, 64) @ (64, 512) = (512, 512)
    # 再做 (512, 512) @ (512, 64) = (512, 64)
    assert scores.shape == (2, 8, 512, 512)
    assert p.shape == (2, 8, 512, 512)
    assert out.shape == (2, 8, 512, 64)


if __name__ == "__main__":
    main()
