import torch


def main() -> None:
    torch.manual_seed(0)

    # 第 3 题：Matmul 基础。
    #
    # @ 是 PyTorch 的矩阵乘法运算符，等价于 torch.matmul(A, B)。
    # 二维矩阵乘法规则：
    #   A.shape = (M, K)
    #   B.shape = (K, N)
    #   C = A @ B
    #   C.shape = (M, N)
    #
    # 中间维度 K 必须相同；结果保留 A 的行数 M 和 B 的列数 N。
    a = torch.randn((2, 3))
    b = torch.randn((3, 4))
    c = a @ b

    print("== Matmul basics ==")
    print("A.shape:", a.shape)
    print("B.shape:", b.shape)
    print("C = A @ B")
    print("C.shape:", c.shape)
    print("Reason: (2, 3) @ (3, 4) = (2, 4)")

    assert c.shape == (2, 4)

    # 手动验证一个元素，帮助理解矩阵乘法不是逐元素相乘。
    # C[0, 0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    manual_c00 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
    torch.testing.assert_close(c[0, 0], manual_c00)

    print("manual C[0, 0]:", manual_c00.item())
    print("torch  C[0, 0]:", c[0, 0].item())
    print("task3 ok")


if __name__ == "__main__":
    main()
