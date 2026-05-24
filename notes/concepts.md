# Concepts - TileLang AI 算子开发概念导读

这份笔记用于在写代码前建立共同词汇。学习 TileLang 时，最重要的不是记住每个 API，而是形成一条清晰链路：

```text
数学表达 -> 张量形状 -> 数据布局 -> 并行分块 -> 内存层级 -> kernel 实现 -> correctness -> benchmark
```

## 1. 什么是 AI 算子

AI 算子是深度学习模型中的一个计算单元，例如 matmul、softmax、layernorm、attention、conv、activation。框架层面你看到的是 PyTorch 函数，底层真正执行的是 GPU kernel。

开发 AI 算子的核心目标通常有三个：

- 正确性：输出必须和 PyTorch reference 在合理误差内一致。
- 性能：减少访存、提高并行度、利用 Tensor Core 或其它硬件单元。
- 可集成性：接口、shape、dtype、layout 要能被上层模型稳定调用。

本工程的毕业目标是 FlashAttention forward，本质是把 attention 的 `QK^T -> softmax -> PV` 融合成一个更省显存、更快的 kernel。

## 2. TileLang 的位置

TileLang 是面向高性能张量程序的 DSL。它让你用 Python 写接近 CUDA/Triton 心智模型的代码，同时通过 TVM/TIR 等编译路径生成 GPU kernel。

可以把 TileLang 理解为三层：

- Python 层：写 kernel factory、传 shape/dtype/config。
- TileLang DSL 层：用 `T.Kernel`、`T.Parallel`、`T.copy`、`T.gemm` 描述并行计算和数据搬运。
- 编译执行层：TileLang JIT 编译成 GPU kernel，再由 PyTorch tensor 作为输入输出。

和 PyTorch 相比，TileLang 要你显式思考线程、分块和内存层级；和 CUDA C++ 相比，TileLang 提供了更高层的 tensor tile、fragment、GEMM intrinsic 表达。

## 3. Kernel、Program、Block、Thread

一个 GPU kernel 是在设备上并行执行的函数。TileLang 里常见入口是：

```python
with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
    ...
```

关键概念：

- `grid`：启动多少个 program/block，每个 block 负责一块输出 tile。
- `threads`：每个 block 中有多少 CUDA threads。
- `bx/by`：当前 block 的坐标，用于定位这块 block 负责的数据范围。
- `T.Parallel`：在 block 内表达并行循环，常用于 elementwise 或 copy 类任务。
- `T.serial` / `T.unroll`：表达串行循环或编译期展开，常用于 reduction、K 维循环、小固定循环。

写 kernel 时要先回答：每个 block 计算输出的哪一块？这个 block 内的 threads 如何分摊搬运和计算？

## 4. Shape、Layout、Stride

张量 shape 描述逻辑维度，例如 attention 中 `q` 的形状是 `(B, H, S, D)`：

- `B`：batch size。
- `H`：attention heads。
- `S`：sequence length。
- `D`：head dimension。

layout 描述数据在内存里的排列方式。PyTorch 默认 contiguous 的 `(B, H, S, D)` 中，最后一维 `D` 连续存放。stride 描述每个维度移动 1 个 index 时内存地址跳多少元素。

算子开发必须关心 layout，因为 GPU 性能高度依赖连续、合并的内存访问。很多性能问题不是数学计算慢，而是数据读写方式不友好。

## 5. 内存层级

GPU 上常见的内存层级可以粗略理解为：

- global memory：显存，容量大，延迟高，是 PyTorch tensor 主要所在位置。
- shared memory：block 内共享，容量小，速度快，适合复用 tile 数据。
- register/local fragment：线程或 warp 附近的临时值，最快但容量最有限。

TileLang 里常见写法：

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
C_local = T.alloc_fragment((block_M, block_N), "float32")
T.copy(A_global_tile, A_shared)
T.gemm(A_shared, B_shared, C_local)
T.copy(C_local, C_global_tile)
```

这体现了一条高性能算子的基本路径：从 global 搬 tile 到 shared，在 fragment/register 中计算，最后写回 global。

## 6. Tiling

tiling 是把大矩阵或大张量拆成小块。它的目的不是改变数学结果，而是让数据更适合 GPU 执行。

以 GEMM `C = A @ B` 为例：

- `block_M/block_N` 决定一个 block 负责的输出矩阵块大小。
- `block_K` 决定每次沿 K 维搬运和累加多少数据。
- 如果 tile 太小，Tensor Core 利用率可能低。
- 如果 tile 太大，shared memory/register 压力可能高，occupancy 可能下降。

本工程的 GEMM 起点使用 `128x128x32` tile，因为这是当前 A100 环境上已经验证正确的 baseline。

## 7. GEMM 为什么重要

GEMM 是 AI 算子开发的核心练习。Linear、MLP、attention 的 `QK^T` 和 `PV` 都可以看成 GEMM 或 batched GEMM。

学习 GEMM 能覆盖几乎所有基础能力：

- block/grid 分解。
- shared memory staging。
- Tensor Core 或 MMA intrinsic。
- fp32 accumulation。
- pipeline copy/compute。
- benchmark 和 autotune。

如果能清楚解释一个 tiled GEMM，就已经掌握了大部分 AI 算子优化的语言。

## 8. Reduction、Softmax 与数值稳定

reduction 是把多个元素合并成一个或少量元素，例如 sum、max、norm。softmax 常见公式是：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

直接计算容易溢出，因此稳定版本会先减去最大值：

```text
m = max(x)
softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
```

Attention 里的 softmax 更难，因为 `S x S` 的 attention matrix 很大。FlashAttention 的关键是 online softmax：一边遍历 K/V block，一边维护 running max `m`、running denominator `l` 和 accumulator `acc`，避免完整物化 attention matrix。

## 9. FlashAttention Forward 的核心思想

普通 attention 计算：

```text
scores = Q @ K^T * scale
probs = softmax(scores)
out = probs @ V
```

问题是 `scores` 和 `probs` 的形状是 `(B, H, S, S)`，当 `S` 很大时显存和带宽成本很高。

FlashAttention forward 的思路：

- 按 Q block 处理输出的一段 query。
- 流式遍历 K/V block。
- 对每个 Q block 维护 `m/l/acc`。
- 每处理一个 K/V block，就更新 softmax 统计量和输出累加器。
- 最终只写出 `(B, H, S, D)` 的结果，不保存完整 `(S, S)` 矩阵。

这也是本工程第 7-9 周的主线。

## 10. Correctness 与 Benchmark

高性能 kernel 的开发顺序应该固定：

1. 写 PyTorch reference。
2. 写最小 kernel。
3. 小 shape correctness。
4. 边界 shape correctness。
5. benchmark。
6. 调 tile/config。
7. 再 benchmark。

不要先追性能再补 correctness。GPU kernel 出错时，错误可能来自边界、layout、dtype、mask、同步、未初始化内存或数值稳定性；没有 reference test 会很难定位。

本工程默认 correctness 对照：

- elementwise/GEMM：对比 PyTorch 原生计算。
- attention：对比 `naive_attention_forward` 和 PyTorch SDPA。
- fp16/bf16：先使用 `rtol=1e-2, atol=1e-2`，再根据误差来源调整。

## 11. 常用术语速查

| 术语 | 含义 |
| --- | --- |
| kernel | GPU 上执行的函数 |
| grid | kernel 启动的 block/program 网格 |
| block/program | 一组 threads 共同执行的计算单元 |
| thread | GPU 中最小的执行线程 |
| warp | NVIDIA GPU 上 32 个 threads 的执行组 |
| tile | 从大张量切出来的一块数据 |
| shared memory | 同一个 block 内共享的高速片上内存 |
| fragment | 存放中间结果的局部 tile，通常用于 MMA/GEMM |
| coalescing | 连续线程访问连续内存，提高访存效率 |
| occupancy | GPU 上活跃 warps/blocks 的占用程度 |
| latency | 单次运行耗时 |
| throughput | 单位时间完成的计算量或数据量 |
| TFLOPS | 每秒万亿次浮点运算 |
| arithmetic intensity | 计算量与访存量的比例 |
| online softmax | 流式更新 softmax 统计量的方法 |

## 12. 建议阅读顺序

1. 先读本文件的 1-6 节，建立 kernel、tile、memory 的基本概念。
2. 跑通 Week 01，确认环境和 vector add。
3. Week 03-04 时回看 5-7 节，理解 GEMM 的数据流。
4. Week 07 前重点复习 8-9 节，准备进入 FlashAttention。
5. 每写一个新算子，都用 `notes/operator_checklist.md` 做自检。

