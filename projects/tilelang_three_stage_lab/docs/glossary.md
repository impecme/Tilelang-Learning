# Glossary

这份术语表用于把英文资料、中文理解和代码中的样子连起来。

| English | 中文 | 在本项目中的样子 |
| --- | --- | --- |
| tensor | 张量 | PyTorch 的 `torch.Tensor` |
| shape | 形状 | `(B, S, hidden)`、`(M, N)` |
| dtype | 数据类型 | `float16`、`bfloat16`、`float32` |
| device | 设备 | `cpu`、`cuda:0` |
| contiguous | 连续布局 | TileLang kernel 默认要求输入连续 |
| reference | 参考实现 | `xxx_reference`，通常是 PyTorch |
| correctness | 正确性 | TileLang 输出和 reference 是否 close |
| tolerance | 误差容忍 | `rtol`、`atol` |
| kernel | GPU 核函数 | TileLang JIT 生成的 GPU 程序 |
| grid | 网格 | 一次 kernel launch 中有多少 block |
| block/program | 线程块 | `T.Kernel(... )` 中一个 program |
| thread | 线程 | `T.Parallel` 中的并行执行单元 |
| tile | 分块 | GEMM 中 `block_M/block_N/block_K` |
| global memory | 全局显存 | PyTorch tensor 所在位置 |
| shared memory | 共享内存 | `T.alloc_shared` |
| fragment | 局部片段 | `T.alloc_fragment` |
| reduction | 归约 | sum、max、norm、softmax |
| GEMM | 通用矩阵乘 | `C = A @ B` |
| fusion | 融合 | `matmul + bias + gelu` |
| logits | 未归一化分数 | LM Head 输出 `(B,S,V)` |

