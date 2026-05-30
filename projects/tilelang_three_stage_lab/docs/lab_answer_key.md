# Full Lab Answer Key

这份文档给五个 full labs 提供预期输出和参考解释。跑 lab 时先自己观察，再用这里校对。

## Basic Kernel Debug

推荐命令：

```bash
python3 scripts/run_lab.py --lab basic
```

预期观察：

- `vector_add/copy/axpy` 都是 elementwise 或一维映射。
- `row_sum` 是教学串行 reduction，正确性优先，不代表高性能。
- shape/device/contiguous 错误应该在 Python guard 阶段被明确抛出。

关键答案：

- boundary guard 保护最后一个不满 block 的 tail。
- `ceildiv(N, block)` 表示覆盖所有元素需要多少个 block。
- reference 的价值是先定义数学语义，再实现 TileLang kernel。

## GEMM Tile Shape

推荐命令：

```bash
python3 scripts/run_lab.py --lab gemm
```

预期观察：

- `gemm_tilelang` v1 拒绝非 tile-aligned shape。
- 错误不是数学限制，而是教学版先避免 tail tile 复杂度。
- `T.copy`、shared memory、fragment、`T.gemm` 是 GEMM 数据流主线。

关键答案：

- `M/N/K` 分别对应输出行、输出列、公共 reduction 维。
- `block_m/block_n` 决定输出 tile 大小。
- `block_k` 决定每次搬运和累加的 K 分块。

## Reduction Optimization

推荐命令：

```bash
python3 scripts/run_lab.py --lab reduction
```

预期观察：

- serial reduction 好读，但一行主要由一个线程工作。
- parallel reduction 用一个 block 的多个线程共同处理一行。
- 小 shape 下 parallel 不一定更快，因为 kernel launch、同步、空线程和 reduction 开销可能抵消收益。

关键答案：

- sum 的 neutral padding 是 `0`。
- max 的 neutral padding 是很小的负数。
- `cols <= block_n <= 1024` 是 v1 单 block reduction 的教学限制。

## Decoder Shape Trace

推荐命令：

```bash
python3 scripts/run_lab.py --lab decoder
```

预期数据流：

```text
x
-> RMSNorm
-> QKV Linear
-> Causal Attention
-> Output Linear
-> Residual
-> RMSNorm
-> MLP(GELU)
-> Residual
-> LM Head
-> logits
```

关键答案：

- QKV linear 把 hidden 投到 `3 * hidden`，再拆成 Q/K/V。
- attention score 的核心 shape 是 `S x S`。
- logits 的 shape 是 `(B, S, vocab_size)`。

## Benchmark Reading

推荐命令：

```bash
python3 scripts/run_lab.py --lab benchmark
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```

预期观察：

- `latency_ms` 是平均耗时，不是总耗时。
- `warmup` 避免把 JIT 编译和 CUDA 初始化计入稳定 latency。
- `notes` 字段要记录限制，比如 teaching kernel、serial reduction、optimized reductions。

关键答案：

- 一次只改一个变量，才能解释变化原因。
- TileLang 版本慢不一定失败；可能是 shape 太小或实现为了教学可读。
- benchmark 结论必须和 correctness 一起看。

