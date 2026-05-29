# Stage 06 - Attention 基线与 Online Softmax

## 阶段目标

这一阶段，我进入毕业项目的数学和 reference 层。目标是彻底理解普通 attention、materialized attention 的显存问题，以及 FlashAttention 使用 online softmax 避免保存完整 attention matrix 的原因。

## 先修状态

- 已完成 Stage 05，或至少理解 GEMM、softmax、fp32 accumulation。
- 已完成 Stage 00.5，能解释 `QK^T -> softmax -> P@V`。
- 能解释 `QK^T` 和 `P@V` 都是 GEMM 类计算。

## 阅读

- `kernels/reference.py`
  - `naive_attention_forward`
  - `online_attention_forward`
  - `sdpa_attention_forward`
- `notes/concepts_deep_dive.md` 第 7、8、9、11 节。
- `notes/stage00_5_transformer_basics.md`，重点复习 Q/K/V、shape、mask。
- `kernels/flash_attention.py`
- `tests/test_flash_attention_reference.py`
- PyTorch `scaled_dot_product_attention` 行为说明。
- FlashAttention 论文或公式讲解。

## 概念

- Q/K/V：query、key、value。
- shape：`(B, H, S, D)`。
- `B`：batch size。
- `H`：num heads。
- `S`：sequence length。
- `D`：head dimension。
- scale：默认 `1 / sqrt(D)`。
- scores：`Q @ K^T * scale`。
- mask：屏蔽不应该看的位置。
- causal mask：第 `i` 个 query 只能看 `j <= i` 的 key。
- probs：`softmax(scores)`。
- output：`probs @ V`。
- materialized attention：显式保存 `(B, H, S, S)` scores/probs。
- online softmax：分块流式更新 softmax 统计量。
- running max `m`：当前已经看过 block 的最大 score。
- denominator `l`：当前 softmax 分母。
- accumulator `acc`：当前输出累加器。
- tolerance：fp16/bf16 结果比较的误差容忍。
- non-divisible S：sequence length 不能被 block size 整除。

## 代码

- `kernels/reference.py`
  - `_check_attention_inputs`：输入检查。
  - `_attention_scale`：scale 处理。
  - `naive_attention_forward`：materialized baseline。
  - `online_attention_forward`：FlashAttention 形状的 PyTorch reference。
  - `sdpa_attention_forward`：PyTorch optimized baseline。
- `benchmarks/bench_flash_attention.py`
  - 三种 baseline 的 benchmark。
- `tests/test_flash_attention_reference.py`
  - fp16/bf16、非整除 shape、causal、SDPA 对照。

## 练习

1. 手写 attention 公式：

   ```text
   scores = Q @ K^T * scale
   probs = softmax(scores)
   out = probs @ V
   ```

2. 手写 online softmax 更新：
   - 旧状态：`m_old, l_old, acc_old`。
   - 当前 block scores：`scores_block`。
   - 当前 block max：`m_block`。
   - 新最大值：`m_new = max(m_old, m_block)`。
   - 旧状态缩放：`old_scale = exp(m_old - m_new)`。
   - 当前概率未归一化：`p = exp(scores_block - m_new)`。
   - 新分母：`l_new = l_old * old_scale + sum(p)`。
   - 新累加器：`acc_new = acc_old * old_scale + p @ V_block`。

3. 跑 benchmark：

   ```bash
   python3 -m benchmarks.bench_flash_attention --batch 1 --heads 1 --seq 128 --dim 64 --warmup 10 --repeat 50
   python3 -m benchmarks.bench_flash_attention --batch 2 --heads 8 --seq 512 --dim 64 --warmup 10 --repeat 50
   ```

4. 画数据流：
   - Q block 固定。
   - K/V block 流式遍历。
   - 每个 block 更新 `m/l/acc`。
   - 最终写出 output block。

## 报告模板

建议写 `reports/stage06_attention_baseline.md`：

```markdown
# Stage 06 Attention Baseline Report

## Attention Formula

## Materialized Memory Cost

## Online Softmax Derivation

## Benchmark Results

## Q/K/V Block Data Flow

## Numerical Tolerance
```

## 思考问题

- 为什么普通 attention 的中间矩阵是 `(B,H,S,S)`？
- 当 `S=4096` 时，scores/probs 的显存压力为什么会很大？
- online softmax 为什么需要同时维护 `m` 和 `l`？
- causal mask 在 block 内如何影响 scores？
- PyTorch SDPA 为什么通常比 naive attention 快？

## 验收标准

- 能解释 FlashAttention 为什么不保存完整 attention matrix。
- 能手写 online softmax 更新公式。
- 确认 `online_attention_forward` 与 naive attention 在核心 shape 上 close。
- 至少有两组 attention benchmark 结果。
