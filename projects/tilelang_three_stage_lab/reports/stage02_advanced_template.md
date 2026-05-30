# Stage 02 Advanced Ops Report

## Stage Goal

- 我是否能解释 GEMM 的 global/shared/fragment/global 数据流：
- 我是否已经阅读 `docs/stages/02_advanced_ops.md`：

## Environment

- Date:
- GPU:
- Python:
- PyTorch:
- CUDA:
- TileLang:

## Commands

```bash
pytest tests/test_advanced.py
RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -m tilelang
python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 scripts/run_lab.py --lab gemm
python3 scripts/run_lab.py --lab reduction
```

## Correctness

| op | shape | dtype | tolerance | result | notes |
| --- | --- | --- | --- | --- | --- |
| gemm | 128x128x128 | fp16 | 1e-2 | | |
| row_softmax | 8x16 | fp32 | 1e-5 | | |
| row_softmax_parallel | 8x16 | fp32 | 1e-5 | | |
| rmsnorm | 8x16 | fp32 | 1e-5 | | |
| rmsnorm_parallel | 8x16 | fp32 | 1e-5 | | |
| gelu | 8x16 | fp32 | 1e-5 | | |
| linear_bias_gelu | 128x128 | fp16 | 2e-2 | | |

## Benchmark Results

| op | torch ms | tilelang ms | notes |
| --- | --- | --- | --- |
| gemm | | | |
| rmsnorm | | | serial reduction |
| rmsnorm_parallel | | | single-block reduction |
| row_softmax_parallel | | | T.reduce_max + T.reduce_sum |
| gelu | | | |

## What I Learned

- shared memory:
- fragment:
- stable softmax:
- RMSNorm:
- parallel reduction:
- tile-aligned GEMM:

## Hands-On Lab Notes

- 我改了什么 shape/tile 参数：
- 遇到的 shape guard 报错：
- 我如何修复：
- benchmark 中最值得解释的一行：
- 下一步只改一个什么变量：

## Stage Gate

- 我能解释 `M/N/K`：
- 我能画出 GEMM 数据流：
- 我能说明 stable softmax 和 RMSNorm 的 reduction：
- 我能解释串行 reduction 和并行 reduction 的区别：
- 我能解释 fp16 tolerance：
- 是否可以进入 Stage 03：

## Open Questions

-
