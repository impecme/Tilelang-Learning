# Stage 05 - Autotuning 与常见 AI 算子模式

## 阶段目标

这一阶段，我把单个 GEMM 练习扩展到“系统搜索配置”和“常见 AI 算子模式”。目标是理解高性能算子不是只写一次代码，而是需要围绕 shape、dtype、layout、tile config 做验证、搜索和记录。

## 先修状态

- 已完成 Stage 04。
- 已有 GEMM benchmark 表格。
- 理解 reduction、fp32 accumulation、softmax 数值稳定。

## 阅读

- TileLang Autotuning。
- `notes/concepts_deep_dive.md` 第 7、8、9、10 节。
- TileLang Elementwise Operators。
- TileLang GEMV/GEMM operator examples。
- PyTorch softmax、layernorm/RMSNorm、activation 的行为说明。
- `benchmarks/utils.py`。
- `notes/operator_checklist.md`。

## 概念

- search space：待搜索的参数组合集合。
- config：一组具体参数，例如 block size、threads、num_stages。
- compile cost：每个配置可能触发一次 JIT 编译。
- cache：避免重复编译相同配置。
- best config：在给定 shape/dtype/hardware 上表现最好的配置。
- fusion：把多个算子合并到一个 kernel 中，减少中间读写。
- matmul+bias+activation：常见 Linear 后处理融合模式。
- row-wise softmax：按行做 max、exp、sum、normalize。
- max-subtraction：softmax 数值稳定技巧。
- layernorm：减均值、除标准差、scale/bias。
- RMSNorm：用 root mean square 归一化，不减均值。
- memory traffic：global memory 读写量。
- arithmetic intensity：计算量和访存量的比例。

## 代码

- `kernels/gemm.py`：已有可调参数。
- `benchmarks/bench_gemm.py`：benchmark 参数入口。
- `kernels/reference.py`：attention reference 中 softmax 和 fp32 cast 的处理。

## 练习

1. GEMM autotune 记录：
   - 定义小搜索空间。
   - 至少搜索 `threads` 和 `num_stages`。
   - 记录每个配置的 latency。
   - 标出 best config。

2. 三选二实现：
   - `matmul + bias + relu` 或 `matmul + bias + gelu`。
   - row-wise softmax。
   - RMSNorm。

3. 每个算子都要包含：
   - PyTorch reference。
   - TileLang 或 correctness-first 实现。
   - correctness test。
   - benchmark smoke。
   - dtype 和 tolerance 说明。

4. 访存量估算：
   - 对一个 fusion 算子估算融合前后的 global memory 读写次数。
   - 写清楚为什么 fusion 可能更快。

## 报告模板

建议写 `reports/stage05_autotune_operator_patterns.md`：

```markdown
# Stage 05 Report

## Autotune Search Space

## GEMM Results

## Best Config

## Operator Pattern 1

## Operator Pattern 2

## Memory Traffic Estimate

## Numerical Notes
```

## 思考问题

- 搜索空间越大一定越好吗？
- 为什么 best config 需要绑定 shape、dtype 和 hardware？
- fusion 为什么能减少 memory traffic？
- softmax 为什么必须做 max-subtraction？
- RMSNorm 和 LayerNorm 的归一化统计量有什么不同？

## 验收标准

- 能复现一个 best config。
- 至少完成两个常见 AI 算子模式。
- 每个完成的算子都有 reference、test、benchmark。
- 能用访存量解释 fusion 为什么可能更快。
- 能说明 fp32 accumulation 对 reduction/softmax 的意义。
