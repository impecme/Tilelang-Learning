# Stage 03：Decoder Block Pipeline

大型综合阶段把前两阶段的算子组合起来，形成一个“单 Decoder Block + logits”的迷你推理流水线。它不是完整 LLM 产品，但足够帮助你理解真实模型中常见的张量流。

## 阶段定位

- 难度：综合。
- 核心文件：`tilelab/decoder.py`。
- 对应测试：`tests/test_decoder.py`。
- 对应 benchmark：`benchmarks/bench_decoder_block.py`。
- 对应 lab：`labs/04_decoder_shape_trace/README.md`。
- 学习报告：`reports/stage03_decoder_template.md`。

## 细化学习目标

- 能解释 `MiniDecoderConfig` 中 `hidden_size`、`num_heads`、`head_dim`、`ffn_hidden_size`、`vocab_size` 的作用。
- 能说明 `hidden_size == num_heads * head_dim` 为什么是硬约束。
- 能从 `x=(B,S,H)` 推导 QKV linear 后的 `q/k/v` shape。
- 能解释 causal attention 中 `scores`、`probs`、`attention_out` 的 shape。
- 能说出两次 RMSNorm 和两次 residual add 的位置。
- 能解释 MLP 的 up projection、GELU、down projection。
- 能说明 LM Head 为什么输出 `logits=(B,S,V)`。
- 能指出 TileLang 路径中哪些 PyTorch 小操作已经被教学 kernel 替换。
- 能对比 `decoder_block_tilelang` 和 `decoder_block_tilelang_optimized` 的 reduction 路径。
- 能用 CSV benchmark 记录 decoder block 的 reference 和 TileLang latency。

## 必会概念

| 概念 | 小白解释 |
| --- | --- |
| Decoder Block | LLM 中重复堆叠的推理模块 |
| QKV Linear | 一次线性层生成 query/key/value |
| head split | 把 hidden 切成多个 attention head |
| causal mask | 让当前位置只能看自己和过去 token |
| residual add | 把模块输入直接加回输出，稳定深层网络 |
| MLP | attention 后的前馈网络 |
| logits | 每个位置对词表中每个 token 的未归一化分数 |
| small fusion | 把 scale、mask 等小操作合并，减少中间 tensor 和 launch |
| optimized reductions | 用并行 RMSNorm/softmax 替换串行教学版 reduction |

## 逐步任务

1. 阅读 `docs/tensor_shapes.md`，先把默认 shape 写下来。
2. 阅读 `docs/kernels/decoder_block.md`，标出每一步输入输出 shape。
3. 阅读 `MiniDecoderConfig` 和权重检查逻辑，理解为什么先验证 shape/dtype/device/contiguous。
4. 阅读 reference 路径，确认 PyTorch 标准答案如何计算。
5. 阅读 TileLang 路径，找出哪些步骤调用了前两阶段的 TileLang 算子。
6. 阅读 `docs/optimization_log.md`，找出 `_linear_tilelang` 和 attention score 路径的优化点。
7. 阅读 `docs/reduction_optimization.md`，找出 optimized 入口替换了哪些 reduction 算子。
8. 运行 `python3 scripts/run_lab.py --lab decoder` 和 `python3 scripts/demo_common_errors.py --case decoder_shape`。
9. 运行 decoder 测试和 tiny TileLang smoke，确认 logits 与 reference close。
10. 运行 decoder benchmark CSV，把完整流水线和性能观察写进报告模板。

## 必须运行

```bash
python3 scripts/check_project.py --stage decoder
python3 scripts/run_lab.py --lab decoder
pytest tests/test_decoder.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_decoder.py -q -m tilelang
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --warmup 1 --repeat 1
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --compare-optimized --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv
```

## 实验记录要求

- 记录一组完整 config。
- 写出 `x -> logits` 的每一步 shape。
- 记录 decoder block 输出和 logits 的 correctness tolerance。
- 记录至少一个 validation 报错案例，例如错误 head 配置或错误权重 shape。
- 记录一次 `/tmp/tilelang_decoder.csv`，并说明 TileLang 路径里哪些小操作已经去 PyTorch 化。
- 对比 `tilelang_decoder_block` 和 `tilelang_decoder_block_optimized_reductions`，说明优化入口只替换了 reduction 路径。

## 验收问题

- 为什么 `hidden_size` 必须等于 `num_heads * head_dim`？
- `QK^T` 的输出为什么是 `(S,S)`？
- causal mask 屏蔽的是未来 token 还是过去 token？
- residual add 为什么需要保留原输入？
- logits 为什么还不是概率？
- `scale_causal_mask_tilelang` 为什么能减少一次中间操作？
- `decoder_block_tilelang_optimized` 为什么不是完整 fused attention？
- `linear_bias_tilelang` 为什么是教学组合版而不是最终融合版？

## 常见卡点

- 把 `(B,H,S,D)` 和 `(B,S,H,D)` 混用。
- 忘记 K 在 attention score 中需要转置。
- causal mask 方向写反。
- 在 TileLang 路径传入非 contiguous tensor。
- 用非教学 shape 触发 GEMM tile 对齐限制。

## 项目完成条件

当你能从 `x=(B,S,H)` 一路追踪到 `logits=(B,S,V)`，并能说明每个 TileLang 算子在流水线中的位置时，本项目的 v1 学习目标就完成了。
