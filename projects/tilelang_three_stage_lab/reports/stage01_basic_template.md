# Stage 01 Basic Kernels Report

## Stage Goal

- 我是否能解释 `T.Kernel`、`T.Parallel`、global index 和 boundary guard：
- 我是否已经阅读 `docs/stages/01_basic_kernels.md`：

## Environment

- Date:
- GPU:
- Python:
- PyTorch:
- CUDA:
- TileLang:

## Commands

```bash
pytest tests/test_basic.py
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -m tilelang
python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 scripts/run_lab.py --lab basic
python3 scripts/demo_common_errors.py --case cpu_tensor
```

## Correctness

| op | shape | dtype | result | notes |
| --- | --- | --- | --- | --- |
| vector_add | N=1024 | fp32 | | |
| vector_add | N=1000 | fp32 | | boundary guard |
| copy | N=1024 | fp32 | | |
| axpy | N=1024 | fp32 | | |
| row_sum | M=8,N=17 | fp32 | | |
| row_sum_parallel | M=8,N=17 | fp32 | | single-block reduction |

## What I Learned

- `T.Kernel`:
- `T.Parallel`:
- boundary guard:
- elementwise vs reduction:
- serial vs parallel row_sum:

## Hands-On Lab Notes

- 我改了什么参数：
- 运行结果：
- 触发过的错误：
- 我如何定位这个错误：
- 下一步只改一个什么变量：

## Stage Gate

- 我能解释 `idx = bx * block_size + i`：
- 我能说明 `N=1000` 为什么需要 boundary guard：
- 我能根据报错区分 shape/dtype/device/contiguous 问题：
- 是否可以进入 Stage 02：

## Open Questions

-
