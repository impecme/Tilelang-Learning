def ceildiv(a: int, b: int) -> int:
    """Return ceil(a / b) for non-negative integer a and positive integer b.

    ceildiv 是向上取整除法。它常用来计算 GPU kernel 需要启动多少个
    block 才能覆盖所有元素。

    例子：
    - 1000 个元素，每个 block 处理 256 个位置。
    - 1000 / 256 = 3.90625。
    - block 数不能是小数，所以需要向上取整为 4。

    公式 `(a + b - 1) // b` 的意思是先把 a 往上补一点，再做整数除法：
    - Python 中 `//` 是整除，结果会去掉小数部分。
    - `a + b - 1` 可以让没有整除的情况多算出 1 个 block。
    """
    if a < 0:
        raise ValueError(f"ceildiv expects a >= 0, got {a}")
    if b <= 0:
        raise ValueError(f"ceildiv expects b > 0, got {b}")
    return (a + b - 1) // b


def main(use_cuda: bool = False) -> None:
    del use_cuda
    examples = [
        (1000, 256),
        (1024, 256),
        (0, 256),
    ]
    for a, b in examples:
        print(f"ceildiv({a}, {b}) = {ceildiv(a, b)}")

    result = ceildiv(1000, 256)
    # assert 用来做简单检查：如果 result == 4 为 True，程序继续运行；
    # 如果 result 不是 4，Python 会抛出 AssertionError，说明结果不符合预期。
    assert result == 4
    print("task1 ok")


if __name__ == "__main__":
    main()
