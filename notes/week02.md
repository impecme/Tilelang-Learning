# Week 02 - TileLang DSL 核心

## Goals

- 熟悉 `T.Kernel`、`T.Parallel`、`T.serial`、`T.unroll`。
- 练习动态 shape 和 dtype 参数。
- 实现 elementwise、copy、简单 reduce 练习。

## Exercises

- 扩展 `vector_add` 为 `axpy`：`out = alpha * x + y`。
- 实现一个 row-wise sum reference 和 TileLang 版本。
- 给每个新算子补 PyTorch correctness test。

## Notes

- 记录每种 loop construct 对应的硬件直觉。
- 记录 TileLang 编译错误时最有帮助的错误信息。

## Done Criteria

- 至少两个 DSL 小练习通过测试。
- 能解释 `T.Parallel` 与直接使用 thread index 的差异。

