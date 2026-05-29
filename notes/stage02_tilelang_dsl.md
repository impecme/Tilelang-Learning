# Stage 02 - TileLang DSL 与基础控制流

## 阶段目标

这一阶段，我开始系统学习 TileLang DSL。目标是能独立写小型 kernel，并知道什么时候用并行循环、串行循环、unroll、shared memory、fragment 和 `out_idx`。

## 先修状态

- 已完成 Stage 01。
- 能解释 vector add 的 thread 分工。
- 能写一个 PyTorch reference test。

## 阅读

- TileLang Language Basics。
- `notes/concepts_deep_dive.md` 第 4、6、7、8、13 节。
- TileLang Programming Guides 中 Control Flow。
- TileLang Type System。
- TileLang Python Compatibility。
- `tests/test_vector_add.py`。
- `kernels/vector_add.py`。

## 概念

- 静态 shape 参数：编译 kernel 时确定的 `M/N/K/block_size`。
- 动态 tensor 输入：运行时传入的 PyTorch tensor。
- dtype：`float16`、`bfloat16`、`float32`。
- accum dtype：中间累加使用的 dtype，常用 `float32`。
- `T.Parallel`：适合表达 block 内互相独立的并行工作。
- `T.serial`：适合表达必须顺序执行的循环。
- `T.unroll`：适合小而固定的循环，提示编译器展开。
- `T.alloc_shared`：分配 shared memory tile。
- `T.alloc_fragment`：分配局部 fragment，常用于 GEMM accumulator。
- `T.copy`：在 global/shared/fragment 之间搬运 tile。
- TileLang kernel factory：Python 函数接收 shape/config，返回 `T.prim_func`。
- JIT cache：同一组 shape/config 可能复用编译结果。
- `out_idx`：告诉 TileLang JIT 哪些参数是输出，由 adapter 分配或处理输出。

## 代码

- `kernels/vector_add.py`
  - 惰性 import TileLang 的方式。
  - `@lru_cache` 缓存编译函数。
  - `@tilelang.jit` 的位置。
- `kernels/gemm.py`
  - `@tilelang.jit(out_idx=[-1])`。
  - dtype object 与 dtype string 的区别。
  - `_compile_gemm` 的 shape/config 参数。

## 练习

1. Copy kernel：
   - 输入 `x`，输出 `y`。
   - 支持一维 contiguous tensor。
   - 加 shape mismatch test。

2. Row-wise sum：
   - 输入 shape `(M, N)`。
   - 输出 shape `(M,)`。
   - PyTorch reference：`x.float().sum(dim=-1)`。
   - 重点观察 reduction 不能简单地让每个元素独立写结果。

3. Row-wise max：
   - 输入 shape `(M, N)`。
   - 输出 shape `(M,)`。
   - PyTorch reference：`x.float().max(dim=-1).values`。
   - 思考初始值应该是什么。

4. 错误案例：
   - 写一次故意的 shape 错误。
   - 写一次 dtype 不支持错误。
   - 记录错误信息、定位方式、修复方式。

## 报告模板

建议写 `reports/stage02_tilelang_dsl.md`：

```markdown
# Stage 02 Report

## DSL Constructs

## Copy Kernel

## Row-wise Sum

## Row-wise Max

## Error Case Study

## Open Questions
```

## 思考问题

- `T.Parallel` 和 Python `for` 循环的含义有什么不同？
- 哪些参数应该作为编译期 config？哪些应该作为 tensor runtime input？
- 为什么 reduction 比 elementwise 更难？
- `out_idx` 给开发体验带来了什么便利？

## 验收标准

- 能解释 `T.Parallel`、`T.serial`、`T.unroll` 的使用边界。
- 至少完成 copy、row-wise sum、row-wise max 中两个练习。
- 为每个完成的练习准备 PyTorch reference 和 correctness test。
- 有一份完整错误案例记录。
