# Reduction Optimization

这篇文档讲本轮新增的“并行 reduction 教学版”。它不是最终生产级写法，而是从串行 reduction 迈向高性能 kernel 的第一步。

## 要解决的问题

旧版教学 kernel 为了容易读，使用 `threads=1`：

```text
一行数据 -> 一个 CUDA program -> 一个线程从左到右累加
```

这种写法非常清楚，但一行里只有一个线程干活。对于 `row_sum`、`row_softmax`、`rmsnorm` 这类每行都要聚合很多列的算子，GPU 的并行能力没有用起来。

本轮新增 parallel 版本：

```text
一行数据 -> 一个 CUDA program -> block_n 个线程共同读这一行 -> T.reduce_* 汇总
```

## 新增接口

| 串行教学版 | 并行教学版 | 学习重点 |
| --- | --- | --- |
| `row_sum_tilelang` | `row_sum_parallel_tilelang` | `T.reduce_sum` |
| `row_softmax_tilelang` | `row_softmax_parallel_tilelang` | `T.reduce_max + T.reduce_sum` |
| `rmsnorm_tilelang` | `rmsnorm_parallel_tilelang` | fp32 square sum |
| `decoder_block_tilelang` | `decoder_block_tilelang_optimized` | Decoder 中替换 reduction 路径 |
| `mini_inference_tilelang` | `mini_inference_tilelang_optimized` | logits 端到端对比 |

reference 版本仍然是 PyTorch 公式，用来保证 correctness。

## Shape 限制

parallel v1 只做单 block reduction：

- 输入必须是 rank-2，按最后一维做 reduction。
- `cols <= block_n`。
- `block_n` 必须是 2 的幂。
- `block_n <= 1024`。

如果不满足，会在 Python 侧提前抛 `ValueError`。这不是数学上不能做，而是 v1 不做跨 block reduction，先把单 block 的数据流讲清楚。

## 数据流

### row_sum

```text
X[row, col] -> values[col] -> T.reduce_sum(values) -> Y[row]
```

当 `col >= N` 时填 `0.0`，因为 `0` 不影响求和。这叫 neutral padding。

### row_softmax

```text
X[row, col] -> values[col]
values -> T.reduce_max -> max_val
exp(X - max_val) -> exps[col]
exps -> T.reduce_sum -> sum_val
exps[col] / sum_val -> Y[row, col]
```

当 `col >= N` 时：

- max 阶段填极小值，避免影响最大值。
- sum 阶段填 `0.0`，避免影响分母。
- 写回阶段用 boundary guard，只写真实列。

### rmsnorm

```text
X[row, col]^2 -> squares[col]
squares -> T.reduce_sum -> ss
scale = rsqrt(ss / N + eps)
X[row, col] * scale * weight[col] -> Y[row, col]
```

中间统计量使用 fp32，输出再回到输入 dtype。

## 怎么跑

Correctness：

```bash
pytest tests/test_basic.py tests/test_advanced.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py tests/test_advanced.py -q -m tilelang
```

Reduction benchmark：

```bash
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```

Decoder before/after：

```bash
python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder_opt.csv
```

## 如何看结果

先比较三组：

- `tilelang_*_serial`
- `tilelang_*_parallel`
- `torch_*`

如果 parallel 比 serial 快，说明这一行的多个列确实被更多线程分担了。如果没有更快，先检查：

- shape 是否太小，kernel launch 开销占主导。
- `block_n` 是否远大于 `cols`，空线程太多。
- dtype 和 tolerance 是否一致。
- 第一次 JIT 编译是否被算进了 latency。

如果 CSV 行很多，先用：

```bash
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```

脚本会自动提示 serial/parallel 的 speedup；当 parallel 更慢时，也会提醒你先考虑小 shape、launch overhead、空线程和同步开销。

## 练习

1. 把 `--cols` 从 `64`、`128`、`256` 分别跑一遍，观察 serial 和 parallel 差距。
2. 把 `--block-n` 从 `128` 改成 `256`，记录空线程对小 shape 的影响。
3. 故意设置 `--cols 300 --block-n 256`，观察报错，并解释为什么 v1 不支持。
4. 跑 decoder `--compare-optimized`，说明 optimized 入口只替换了哪些算子。

## 下一步

更高阶的优化方向包括：

- 跨 block reduction，支持 `cols > 1024`。
- warp-level reduction，减少 shared/fragment 开销。
- 把 softmax 和 attention 的 `P@V` 融合得更深。
- 对不同 shape 做 autotune，自动选择 `block_n`。
