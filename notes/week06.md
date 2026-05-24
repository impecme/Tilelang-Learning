# Week 06 - AI 算子常见模式

## Goals

- 实现至少两个常见模式：matmul+bias+activation、softmax、layernorm/RMSNorm。
- 重点练习数值稳定和融合收益分析。

## Exercises

- 实现 `matmul + bias + relu` 或 `matmul + bias + gelu`。
- 实现 row-wise softmax reference 和 TileLang 版本。
- 可选实现 RMSNorm。

## Notes

- 记录 fp32 accumulation 的必要性。
- 记录融合前后内存读写量估算。

## Done Criteria

- 至少两个算子通过 correctness tests。
- 有一份融合收益的定性或定量分析。

