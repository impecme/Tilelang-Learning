# Week 08 - FlashAttention Forward v1

## Goals

- 实现 non-causal TileLang FlashAttention forward v1。
- 不物化完整 attention matrix。
- 保持 `flash_attention_forward(q, k, v, causal=False, sm_scale=None)` 接口不变。

## Exercises

- 将 Q 按 block_M 分块，K/V 按 block_N 流式遍历。
- 用 fp32 保存 `m/l/acc`。
- 先支持 `D=64`，再扩展到 `D=128`。

## Notes

- 记录每个 block 的 shared/fragment 使用量。
- 记录 correctness 失败时的最大误差和对应 shape。

## Done Criteria

- `(1, 1, 128, 64)` non-causal correctness 通过。
- 非 block 整除的 `S` shape 通过。

