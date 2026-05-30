import pytest
import torch


def ceildiv(a: int, b: int) -> int:
    """Return ceil(a / b) for positive integers."""
    return (a + b - 1) // b


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return x.float().sum(dim=-1)


def test_ceildiv() -> None:
    # pytest 会自动执行 test_ 开头的函数。
    assert ceildiv(1000, 256) == 4
    assert ceildiv(1024, 256) == 4
    assert ceildiv(257, 256) == 2


def test_row_sum_reference() -> None:
    x = torch.randn((4, 8), dtype=torch.float16)
    actual = row_sum_reference(x)
    expected = x.float().sum(dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_row_sum_reference_rejects_wrong_rank() -> None:
    # pytest.raises 用来检查某段代码确实会抛出指定异常。
    with pytest.raises(ValueError):
        row_sum_reference(torch.randn((2, 3, 4)))


def main() -> None:
    # 直接运行本文件时，调用 pytest 跑当前文件里的测试。
    exit_code = pytest.main([__file__])
    if exit_code != 0:
        raise SystemExit(exit_code)
    print("task7 ok")


if __name__ == "__main__":
    main()
