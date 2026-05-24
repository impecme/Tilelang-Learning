# Week 05 - Autotuning 与 Benchmark 工程化

## Goals

- 使用 TileLang autotune 搜索配置。
- 固化 benchmark 输出格式。
- 建立可复现实验记录习惯。

## Exercises

- 给 GEMM 增加 autotune 配置空间。
- 记录 warmup、repeat、shape、dtype、GPU 型号、TileLang 版本。
- 把最佳结果保存到 `reports/week05_autotune_results.csv`。

## Notes

- 记录哪些参数最影响性能。
- 记录 autotune 编译时间和缓存行为。

## Done Criteria

- 能复现同一 shape 的最佳配置。
- benchmark 结果包含足够元数据，不依赖记忆解释。

