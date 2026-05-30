import torch


def main() -> None:
    torch.manual_seed(0)

    # 第 4 题：Attention shape。
    #
    # Attention 中常见输入 shape 是 (B, H, S, D)：
    #   B = batch size
    #   H = attention head 数量
    #   S = sequence length，也就是 token 数量
    #   D = head dimension，也就是每个 token 向量的维度
    q = torch.randn((2, 8, 16, 64))
    k = torch.randn((2, 8, 16, 64))
    v = torch.randn((2, 8, 16, 64))

    # k.transpose(-2, -1) 只交换最后两个维度：
    #   (B, H, S, D) -> (B, H, D, S)
    #
    # 这样 q @ k_t 的最后两个维度才能做矩阵乘法：
    #   (S, D) @ (D, S) = (S, S)
    k_t = k.transpose(-2, -1)
    scores = q @ k_t

    print("== Attention shape ==")
    print("Q.shape:", q.shape)
    print("K.shape:", k.shape)
    print("V.shape:", v.shape)
    print("K.transpose(-2, -1).shape:", k_t.shape)
    print("scores = Q @ K.transpose(-2, -1)")
    print("scores.shape:", scores.shape)

    # 对每个 batch/head 来说：
    #   Q[b, h].shape = (16, 64)
    #   K[b, h].T.shape = (64, 16)
    #   scores[b, h].shape = (16, 16)
    #
    # 前面的 (B, H) 是 batch 维，保留下来，所以整体是 (2, 8, 16, 16)。
    assert k_t.shape == (2, 8, 64, 16)
    assert scores.shape == (2, 8, 16, 16)

    print("Reason: (B,H,S,D) @ (B,H,D,S) = (B,H,S,S)")
    print("task4 ok")


if __name__ == "__main__":
    main()
