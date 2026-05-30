import torch


def main() -> None:
    torch.manual_seed(0)

    # 第 5 题：Broadcasting。
    #
    # Broadcasting 是 PyTorch 自动扩展维度的规则。
    # 核心规则：
    #   1. 从最后一维开始对齐。
    #   2. 两个维度相等，可以广播。
    #   3. 其中一个维度是 1，可以扩展到另一个维度。
    #   4. 缺失的前置维度可以当作 1。
    #   5. 如果这些条件都不满足，就报错。
    x = torch.randn((2, 3))
    a = torch.randn(3)
    b = torch.randn(2, 1)

    y_a = x + a
    y_b = x + b

    print("== Broadcasting ==")
    print("x.shape:", x.shape)
    print("a.shape:", a.shape)
    print("(x + a).shape:", y_a.shape)
    print("Explanation: a.shape=(3,) is treated like (1,3), then expanded to (2,3).")

    print("b.shape:", b.shape)
    print("(x + b).shape:", y_b.shape)
    print("Explanation: b.shape=(2,1), last dim 1 expands to 3.")

    assert y_a.shape == (2, 3)
    assert y_b.shape == (2, 3)

    # 失败例子：
    #   x.shape   = (2, 3)
    #   bad.shape =    (2)
    #
    # 从最后一维对齐时，3 和 2 不相等，也没有任何一个是 1，
    # 所以不能 broadcast。
    bad = torch.randn(2)
    try:
        _ = x + bad
    except RuntimeError as error:
        print("\n== Expected broadcasting error ==")
        print("x + torch.randn(2) failed because last dim is 3 vs 2.")
        print("PyTorch error:", error)
    else:
        raise AssertionError("Expected broadcasting failure, but x + bad succeeded.")

    print("task5 ok")


if __name__ == "__main__":
    main()
