# Stage 01 - GPU 算子基本心智模型

## 阶段目标

这一阶段，我建立写 GPU 算子的基本脑内地图：一个数学表达如何变成 tensor 访问、block 分工、thread 并行、memory movement 和 output writeback。目标不是追求性能，而是能解释一个最小 kernel 是如何运行的。

## 先修状态

- 已完成 Stage 00。
- 已完成 Stage 00.25，能读懂基本 Python/PyTorch 代码。
- 能跑 `pytest` 和 TileLang smoke。
- 知道 PyTorch tensor 有 shape、dtype、device。

## 阅读

- `notes/concepts.md` 第 1-6 节。
- `notes/concepts_deep_dive.md` 第 0-6 节。
- TileLang Language Basics 中 kernel、buffer、loop、programming model 相关部分。
- `kernels/vector_add.py`。
- `tests/test_vector_add.py`。

## 概念

- AI 算子：模型里的一个计算单元，例如 add、matmul、softmax、attention。
- kernel：GPU 上执行的函数。
- grid：kernel 启动时的 block/program 网格。
- block/program：一组 threads 共同完成一块输出。
- thread：GPU 执行的基本线程。
- warp：NVIDIA GPU 上 32 个 threads 的执行组。
- tensor：带 shape、dtype、device、stride 的多维数据。
- shape：逻辑维度。
- stride：每个维度移动一个 index 时，底层地址移动多少元素。
- layout：数据在内存中的排列方式。
- contiguous：张量按默认连续布局存储。
- tile：从大张量切出来的一块数据。
- boundary guard：处理 `N` 不是 block size 整数倍的边界逻辑。
- coalesced access：相邻 threads 访问相邻地址，提高 global memory 效率。
- `@tilelang.jit`：把 Python kernel factory 编译成可执行 kernel。
- `T.prim_func`：TileLang/TIR 风格的 kernel 函数定义。
- `T.Kernel`：声明 grid 和 block 内 threads。
- `T.Parallel`：表达 block 内并行循环。

## 代码

- `kernels/vector_add.py`
  - `_dtype_name`：PyTorch dtype 到 TileLang dtype 字符串的映射。
  - `vector_add_reference`：PyTorch reference。
  - `_compile_vector_add`：TileLang kernel 编译入口。
  - `vector_add_tilelang`：对外调用接口。
- `tests/test_vector_add.py`
  - reference test。
  - optional TileLang smoke test。

## 练习

1. 用文字画出 vector add 数据流：

   ```text
   A[i] + B[i] -> C[i]
   ```

   写清楚：
   - 一个 block 处理多少元素。
   - `bx` 表示什么。
   - `i` 表示什么。
   - `idx = bx * block_size + i` 如何定位全局元素。

2. 写一份术语表，至少包含：
   - kernel
   - grid
   - block/program
   - thread
   - warp
   - tile
   - shared memory
   - fragment

3. 做扩展练习：实现 `axpy = alpha * x + y`。
   - 先写 PyTorch reference。
   - 再写 TileLang kernel。
   - 支持 `float32`，可选支持 `float16`。
   - 增加 correctness test。

4. 做边界练习：
   - 让 `N=1000`，block size 仍是 `256`。
   - 观察是否需要 boundary guard。
   - 如果现有 kernel 对非整除 shape 不安全，记录原因。

## 报告模板

建议写 `reports/stage01_gpu_operator_model.md`：

```markdown
# Stage 01 Report

## Terms

## Vector Add Data Flow

## AXPY Implementation Notes

## Correctness Results

## Questions
```

## 思考问题

- 为什么一维 elementwise 算子适合作为第一个 TileLang kernel？
- vector add 的性能瓶颈更可能是计算还是访存？
- 如果输入 tensor 不是 contiguous，当前 kernel 会发生什么？
- 为什么 boundary guard 是写通用 kernel 时绕不开的问题？

## 验收标准

- 能解释 vector add 的 grid/block/thread 分工。
- 能说明 `idx = bx * block_size + i` 的含义。
- 完成 AXPY reference 和测试。
- 能说出 elementwise kernel 与 matmul kernel 的复杂度差异。
