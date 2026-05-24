# Week 07 - Attention 基线

## Goals

- 写清楚 attention reference。
- 理解 online softmax 的 `m/l/acc` 状态更新。
- 明确 FlashAttention kernel 的 block 分解。

## Exercises

- 阅读 `kernels/reference.py` 中的 `online_attention_forward`。
- 对比 `naive_attention_forward`、`online_attention_forward`、PyTorch SDPA。
- 画出 Q block 和 K/V block 的数据流。

## Notes

- 记录 causal mask 对 block 内计算的影响。
- 记录非整除 sequence length 的处理策略。

## Done Criteria

- 能手写 online softmax 更新公式。
- `python3 -m benchmarks.bench_flash_attention --seq 512 --dim 64` 能跑通。

