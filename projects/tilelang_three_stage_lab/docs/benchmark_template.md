# Benchmark Template

每次 benchmark 建议记录下面信息。

````markdown
# Benchmark Report

## Environment

- Date:
- GPU:
- Python:
- PyTorch:
- CUDA:
- TileLang:

## Command

```bash
python3 -m benchmarks.bench_advanced --run-tilelang --warmup 10 --repeat 50 --csv /tmp/tilelang_advanced.csv
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 10 --repeat 50 --csv /tmp/tilelang_reductions.csv
```

## Results

| suite | op | backend | shape | dtype | warmup | repeat | latency ms | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advanced | torch_gemm | torch | M=128,N=128,K=128 | fp16 | 10 | 50 | | |
| advanced | tilelang_gemm | tilelang | M=128,N=128,K=128 | fp16 | 10 | 50 | | |
| optimization | tilelang_scale_causal_mask | tilelang | S=128,S=128 | fp16 | 10 | 50 | | |
| reductions | tilelang_row_softmax_serial | tilelang | M=128,N=128,block_n=256 | fp16 | 10 | 50 | | serial reduction |
| reductions | tilelang_row_softmax_parallel | tilelang | M=128,N=128,block_n=256 | fp16 | 10 | 50 | | parallel reduction |

## Interpretation

- 哪个版本更快：
- 可能瓶颈：
- 下一次只改变一个变量：
- CSV 文件路径：
- `scripts/summarize_bench_csv.py` 输出摘要：
- `notes` 中需要补充的限制：
````

注意：benchmark 是学习工具。先保证 correctness，再看 latency。

## CSV 快速模板

可以从 `reports/perf_optimization_template.csv` 复制字段。benchmark 脚本会写出相同字段：

```text
suite,name,backend,shape,dtype,latency_ms,warmup,repeat,notes
```

生成 CSV 后可以先跑：

```bash
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```
