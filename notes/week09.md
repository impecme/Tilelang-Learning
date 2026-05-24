# Week 09 - 性能调优与可选 Causal

## Goals

- 优化 block size、threads、num_stages。
- 对比 naive attention、PyTorch SDPA、TileLang FlashAttention。
- 可选加入 causal mask。

## Exercises

- 跑计划中的核心 shape：`(1,1,128,64)`、`(2,8,512,64)`、`(1,16,1024,128)`。
- 记录每个 shape 的 latency 和相对 PyTorch SDPA 的差距。
- 尝试 causal correctness。

## Notes

- 记录瓶颈：内存带宽、occupancy、寄存器压力或同步开销。
- 记录下一轮优化方向。

## Done Criteria

- non-causal 版本比 materialized naive attention 更快。
- 能解释与 PyTorch SDPA 的主要差距。

