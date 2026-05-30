import torch


def show_tensor_info(name: str, x: torch.Tensor) -> None:
    # 这些字段是后续读 TileLang/PyTorch 测试时最常查的元信息。
    #
    # print(f"{name}.shape:", x.shape) 是 f-string 写法：
    # - f"{name}.shape:" 会把变量 name 的值填进字符串。
    # - 如果 name == "x"，最终打印出来的前半段就是 "x.shape:"。
    print(f"{name}.shape:", x.shape)
    print(f"{name}.ndim:", x.ndim)
    print(f"{name}.dtype:", x.dtype)
    print(f"{name}.device:", x.device)
    print(f"{name}.is_cuda:", x.is_cuda)

    # stride 通常翻译为“步幅”或“跨度”。
    # x.stride() 返回一个 tuple，表示每个维度前进 1 格时，底层存储地址要跨过
    # 多少个元素。x.stride() 不填参数；如果只想看某一维，可以用
    # x.stride()[dim] 从返回的 tuple 里取。
    print(f"{name}.stride():", x.stride())

    # is_contiguous 表示 tensor 的逻辑顺序和底层内存排列是否连续一致。
    # transpose/permute 之后经常会变成非 contiguous。
    print(f"{name}.is_contiguous():", x.is_contiguous())

    # requires_grad 表示 PyTorch autograd 是否需要追踪这个 tensor 的梯度。
    # 本阶段主要写 reference/test，通常不需要梯度，所以默认是 False。
    print(f"{name}.requires_grad:", x.requires_grad)


def main(use_cuda: bool = False) -> None:
    torch.manual_seed(0)

    print("== Create x ==")
    x = torch.randn((2, 3, 4))
    show_tensor_info("x", x)

    print("\n== Common tensor creation functions ==")
    zeros = torch.zeros((2, 3))
    ones = torch.ones((2, 3))
    empty = torch.empty((2, 3))
    from_python = torch.tensor([[1, 2, 3], [4, 5, 6]])

    print("zeros:\n", zeros)
    print("ones:\n", ones)
    print("empty, uninitialized values:\n", empty)
    print("from_python:\n", from_python)

    print("\n== Device conversion ==")
    if use_cuda and torch.cuda.is_available():
        x_cuda = x.to("cuda")
        print("x.device:", x.device)
        print("x_cuda.device:", x_cuda.device)
    elif use_cuda:
        raise RuntimeError("CUDA is not available, cannot run task2 with --cuda")
    else:
        print("CPU-only run. Add --cuda in the runner if you want to inspect x.to('cuda').")

    print("\n== Dtype conversion ==")
    x_half = x.to(dtype=torch.float16)
    print("x.dtype:", x.dtype)
    print("x_half.dtype:", x_half.dtype)
    print("x dtype is unchanged:", x.dtype == torch.float32)

    print("\n== Transpose and contiguous ==")
    x_transposed = x.transpose(-2, -1)
    print("x_transposed.shape:", x_transposed.shape)
    print("x_transposed.stride():", x_transposed.stride())
    print("x_transposed.is_contiguous():", x_transposed.is_contiguous())

    x_contiguous = x_transposed.contiguous()
    print("x_contiguous.shape:", x_contiguous.shape)
    print("x_contiguous.stride():", x_contiguous.stride())
    print("x_contiguous.is_contiguous():", x_contiguous.is_contiguous())
    print("task2 ok")


if __name__ == "__main__":
    main()
