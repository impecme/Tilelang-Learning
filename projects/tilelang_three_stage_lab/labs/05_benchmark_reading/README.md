# Lab：Benchmark Reading

## 目标

学会读 benchmark CSV：不只看 latency，还要看 shape、dtype、backend、notes。

## 要读的代码

1. `tilelab/common.py::BenchmarkResult`
2. `tilelab/common.py::write_benchmark_csv`
3. `benchmarks/bench_reductions.py`
4. `benchmarks/bench_decoder_block.py`
5. `docs/performance_guide.md`

## 运行命令

```bash
python3 scripts/run_lab.py --lab benchmark
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

## 动手改法

- 打开 `/tmp/tilelang_reductions.csv`，找出 `serial` 和 `parallel`。
- 写下最快和最慢的两行。
- 解释一条 `notes` 字段。

## 自测问题

1. 为什么第一次 JIT 编译不能算稳定 latency？
2. memory-bound 和 compute-bound 怎么粗略区分？
3. 为什么 benchmark 前必须先做 correctness？

## 常见错误

- 不记录 shape，导致结果不可复现。
- 忽略 `repeat` 太小带来的波动。
- 只比较 PyTorch 和 TileLang，不解释差异来源。
