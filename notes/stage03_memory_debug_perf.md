# Stage 03 - 内存层级、调试与性能观察

## 阶段目标

这一阶段，我把注意力从“能写出来”转到“能定位问题、能观察性能”。我需要理解数据在 global/shared/fragment 之间如何移动，并能区分编译错误、运行错误、结果错误和性能问题。

## 先修状态

- 已完成 Stage 02，或至少完成 reduction 练习。
- 已经写过 PyTorch reference 与 correctness test。

## 阅读

- TileLang Debugging Tile Language Programs。
- `notes/concepts_deep_dive.md` 第 5、8、9、13 节。
- TileLang Logging/TVM debugging 相关文档。
- `notes/operator_checklist.md`。
- `benchmarks/utils.py`。

## 概念

- global memory 生命周期：kernel 外部的 PyTorch tensor，容量大、访问慢。
- shared memory 生命周期：block 内共享，容量小、访问快。
- local/register：线程私有或局部临时值。
- fragment：TileLang 中常用于 tile 级计算的局部存储。
- bank conflict：shared memory 访问冲突导致性能下降。
- coalescing：global memory 连续访问，提高吞吐。
- alignment：地址对齐对加载效率和 Tensor Core 路径可能有影响。
- IR/lowering：高层 DSL 被降到更底层表示的过程。
- 编译失败：DSL/type/shape/lowering 不合法。
- 运行失败：kernel 启动或执行时出错。
- 结果错误：能跑完但输出不对。
- 性能差：输出正确但 latency 或吞吐不好。
- CUDA event timing：用 CUDA event 统计 GPU 运行时间。
- warmup/repeat：避免首次编译、cache、调度波动影响结果。
- latency：单次运行时间。
- throughput：单位时间完成的数据或计算量。
- TFLOPS：浮点计算吞吐。
- arithmetic intensity：计算量与访存量之比。

## 代码

- `benchmarks/utils.py`
  - `cuda_time_ms`。
  - warmup 和 repeat。
  - `torch.cuda.synchronize()`。
- `benchmarks/bench_gemm.py`
- `benchmarks/bench_flash_attention.py`
- `notes/operator_checklist.md`

## 练习

1. 给 row-wise reduction 加 benchmark：
   - shape 至少包含 `(1024, 1024)` 和 `(4096, 1024)`。
   - 记录 PyTorch baseline。
   - 如果有 TileLang 实现，记录 TileLang latency。

2. 最大误差定位：
   - 对一个错误或近似结果，打印 max abs diff。
   - 找到对应 index。
   - 打印该 index 的 actual、expected、输入片段。

3. Debug 案例记录：
   - 选择一个真实或故意制造的问题。
   - 写清楚现象、初始假设、排查步骤、修复、验证。

4. Benchmark 元数据：
   - 每条 benchmark 结果都记录 shape、dtype、device、GPU、TileLang version、warmup、repeat。

## 报告模板

建议写 `reports/stage03_memory_debug_perf.md`：

```markdown
# Stage 03 Report

## Benchmark Setup

## Reduction Benchmark

## Debug Case

## Max Error Analysis

## Lessons
```

## 思考问题

- 为什么 benchmark 前必须 warmup？
- 为什么 CPU timer 不适合直接测 CUDA kernel？
- 结果错误时，先看最大误差还是平均误差？
- shared memory 一定会提升性能吗？

## 验收标准

- 能区分编译失败、运行失败、结果错误、性能差。
- 至少完成一份 debug case study。
- benchmark 结果包含足够元数据。
- 能解释 CUDA event timing 的基本流程。
