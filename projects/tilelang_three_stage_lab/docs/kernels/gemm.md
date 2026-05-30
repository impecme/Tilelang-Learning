# gemm

## 功能

`gemm` 是通用矩阵乘，是 AI 算子学习的核心训练场。

## 公式

```text
C[M, N] = A[M, K] @ B[K, N]
C[m, n] = sum_k A[m, k] * B[k, n]
```

## 输入输出 Shape

```text
A.shape = (M, K)
B.shape = (K, N)
C.shape = (M, N)
```

v1 TileLang kernel 要求：

```text
M % block_m == 0
N % block_n == 0
K % block_k == 0
```

默认 `block_m=128, block_n=128, block_k=32`。

## PyTorch Reference

代码入口：`tilelab/advanced.py::gemm_reference`。

```python
return a @ b
```

## TileLang 数据流

```text
A/B global tile -> shared memory -> fragment accumulator -> C global
```

每个 block 负责一个 `C` tile。沿 K 维分块循环，不断累加。

## 关键代码语义

- `T.alloc_shared`：为 A/B tile 分配 shared memory。
- `T.alloc_fragment`：为 C tile 分配局部累加器。
- `T.clear(C_local)`：累加前清零。
- `T.Pipelined(...)`：表达 K 维分块循环。
- `T.gemm(A_shared, B_shared, C_local)`：执行 tile 级矩阵乘累加。

## 边界条件

本项目 v1 不实现 tail tile guard，所以不对齐 shape 会抛 `ValueError`。这是为了避免 silent 越界，也让学习阶段更聚焦。

## 常见错误

- 把 `B` 传成 `(N, K)`。
- 忘记 `a.shape[1] == b.shape[0]`。
- 用非对齐 shape 调教学 GEMM。
- 没理解 `block_K` 是沿 reduce 维切块。

## 练习题

1. 写出 `M=128,N=128,K=128` 的计算量 `2*M*N*K`。
2. 解释为什么 `C_local` 需要 fp32 accumulation。
3. 把 `block_n` 改成 64，思考输出 tile 变成什么形状。

