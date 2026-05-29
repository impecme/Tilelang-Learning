# TileLang AI 算子开发 Checklist

## 1. Reference

- 明确输入输出 shape、dtype、layout。
- 写 PyTorch reference。
- 写小 shape correctness test。

## 2. TileLang Kernel

- 固定 public API，不让 benchmark 和 tests 依赖临时函数。
- 先做 correctness-first kernel，再做性能优化。
- 明确 global/shared/fragment 中每块数据的生命周期。

## 3. Numerics

- 明确 accumulation dtype。
- 对 softmax、norm、reduction 类算子使用稳定公式。
- 记录 tolerance 选择原因。

## 4. Performance

- 记录理论读写量或 FLOPs。
- benchmark 固定 warmup、repeat、shape、dtype、GPU、TileLang 版本。
- 每次只改变少量配置，避免无法解释结果。

## 5. Debugging

- 先让小 shape 通过，再上大 shape。
- 遇到编译错误时先看 TileLang IR/lowering 信息。
- 遇到结果错误时先定位最大误差所在 index。

## 6. Release Criteria

- correctness 覆盖正常 shape、边界 shape、非整除 shape。
- benchmark 至少包含 PyTorch reference 和 PyTorch optimized baseline。
- 在 README 或 report 中说明已知限制和下一步计划。
