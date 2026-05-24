# Week 01 - 环境与最小闭环

## Goals

- 安装并确认 `tilelang==0.1.9`。
- 跑通 `scripts/check_env.py` 和 `scripts/week01_smoke.py`。
- 理解项目结构：`kernels/`、`tests/`、`benchmarks/`、`reports/`。
- 阅读 `notes/concepts.md` 的 1-6 节，建立 kernel、tile、memory 的基本概念。

## Exercises

- 阅读 `kernels/vector_add.py`，理解 `@tilelang.jit`、`T.prim_func`、`T.Kernel`。
- 用自己的话总结 `kernel`、`grid`、`thread`、`tile`、`shared memory` 五个词。
- 运行默认测试：`pytest`。
- 可选运行 TileLang 编译 smoke：`RUN_TILELANG_SMOKE=1 pytest -m tilelang`。

## Notes

- 记录安装问题、CUDA/PyTorch/TileLang 版本。
- 记录第一次 kernel 编译耗时和缓存行为。
- 记录 TileLang、CUDA、Triton 在表达 kernel 时最明显的一个差异。

## Done Criteria

- `python3 scripts/check_env.py` 能显示 torch、CUDA、tilelang 信息。
- `pytest` 通过。
- 至少写下一个 TileLang DSL 和 CUDA/Triton 的差异点。
