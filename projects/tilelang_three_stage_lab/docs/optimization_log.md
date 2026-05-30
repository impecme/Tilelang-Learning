# Optimization Log

这份日志记录本轮“教学稳妥优化”的思路。它的目标是让你看到一次优化从发现问题到验证结果的完整路径。

## 背景

完整版项目原本已经能跑通 correctness tests 和 benchmark smoke，但 Decoder TileLang 路径里仍混有 PyTorch 小操作：

- GEMM 后 `out + bias`
- attention score 的 `x * scale`
- causal mask 的单独处理

这些操作会增加中间 tensor 和 kernel launch。对教学项目来说，它们是很好的第一批优化对象：实现简单、风险低、容易观察。

## 新增接口

| 接口 | 学习目的 | 说明 |
| --- | --- | --- |
| `scale_tilelang` | elementwise scale | 替代 `x * scale` |
| `add_bias_tilelang` | row-wise bias add | 替代 `x + bias` |
| `linear_bias_tilelang` | 组合式 linear | `gemm_tilelang + add_bias_tilelang` |
| `scale_causal_mask_tilelang` | 小 fusion | 把 score scale 和 causal mask 合在一个 kernel |
| `row_sum_parallel_tilelang` | 并行 reduction | 一个 block 处理一行，使用 `T.reduce_sum` |
| `row_softmax_parallel_tilelang` | 并行 stable softmax | 使用 `T.reduce_max + T.reduce_sum` |
| `rmsnorm_parallel_tilelang` | 并行 RMSNorm | 使用 fp32 square sum |
| `decoder_block_tilelang_optimized` | Decoder 对照入口 | RMSNorm 和 softmax 改用并行 reduction |

## Decoder 改动

原路径：

```text
gemm -> PyTorch bias add
gemm(Q,K) -> PyTorch scale -> TileLang causal mask -> softmax
```

新路径：

```text
gemm -> TileLang add_bias
gemm(Q,K) -> TileLang scale_causal_mask -> softmax
```

非 causal 路径使用：

```text
gemm(Q,K) -> TileLang scale -> softmax
```

并行 reduction 对照路径：

```text
rmsnorm_tilelang -> rmsnorm_parallel_tilelang
row_softmax_tilelang -> row_softmax_parallel_tilelang
decoder_block_tilelang -> decoder_block_tilelang_optimized
```

## 为什么这不是最终形态

真正高性能实现通常会把 bias、activation、scale、mask 做进更大的 fused kernel 或 GEMM epilogue。本项目 v1 先用独立 TileLang teaching kernels，原因是：

- 代码更短，适合阅读。
- correctness 更容易验证。
- benchmark 更容易拆解每个小操作。
- 后续可以继续把它们融合得更深。

## 如何验证

Correctness：

```bash
pytest -q
RUN_TILELANG_SMOKE=1 pytest -q -m tilelang
```

Benchmark：

```bash
python3 -m benchmarks.bench_optimization --run-tilelang --warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_decoder.csv
```

## 预期收益

- 对小操作：减少 PyTorch 调度参与，让数据流更接近 TileLang 编排。
- 对学习：能把 `scale`、`bias add`、`mask` 看成可独立 benchmark 的 kernel。
- 对 Decoder：让 TileLang 路径更一致，方便后续继续融合。
- 对 reduction：能直接比较串行教学版和并行教学版的 latency。
- 对学习记录：CSV 解读脚本会把 fastest/slowest、serial/parallel、baseline/optimized 自动列出来，减少手工看错行的概率。

## 下一步优化方向

1. 支持跨 block reduction，让 `cols > 1024` 的行也能并行处理。
2. 把 `linear_bias_tilelang` 变成真正 GEMM epilogue fusion。
3. 减少 Decoder attention 中 batch/head Python 循环。
4. 把 softmax 与 attention 的 `P@V` 融合得更深。
5. 用 autotune 或 sweep 比较 GEMM/reduction tile 配置。
