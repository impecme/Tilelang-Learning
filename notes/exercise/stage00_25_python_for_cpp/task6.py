import torch


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise sum with fp32 accumulation.

    这个函数是一个很小的 PyTorch reference：
    - 输入必须是 rank-2 tensor，也就是 shape 类似 (M, N)。
    - x.float() 先把输入转成 float32，提高 sum 的数值稳定性。
    - sum(dim=-1) 沿最后一维求和，所以输出 shape 是 (M,)。
    """
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return x.float().sum(dim=-1)


def main() -> None:
    torch.manual_seed(0)

    x = torch.randn((4, 8), dtype=torch.float16)
    actual = row_sum_reference(x)
    expected = x.float().sum(dim=-1)

    print("x.shape:", x.shape)
    print("actual.shape:", actual.shape)
    print("expected.shape:", expected.shape)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print("row_sum_reference correctness passed.")

    # 错误输入示例：rank-3 tensor 不符合 row_sum_reference 的约定。
    bad = torch.randn((2, 3, 4))
    try:
        row_sum_reference(bad)
    except ValueError as error:
        print("rank error caught as expected:", error)
    else:
        raise AssertionError("Expected ValueError for rank-3 input.")


if __name__ == "__main__":
    main()
