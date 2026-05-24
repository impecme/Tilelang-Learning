# Week 04 - GEMM 性能主线

## Goals

- 使用 `T.gemm` 和 fragment accumulator。
- 使用 `T.Pipelined` 组织 copy/compute。
- 对比 PyTorch matmul 的 latency 和 TFLOPS。

## Exercises

- 跑 `python3 -m benchmarks.bench_gemm --run-tilelang`。
- 扫描 `block_m/block_n/block_k/threads/num_stages` 的少量组合。
- 先以 `block_m=128, block_n=128, block_k=32` 作为正确性 baseline。
- 保存结果到 `reports/week04_gemm_a100.md`。

## Notes

- 记录最佳配置和性能瓶颈猜测。
- 记录和 Triton 写法的主要差异。

## Done Criteria

- 有至少 3 组 GEMM 配置的 benchmark 结果。
- 能解释 shared/fragment/pipeline 在 GEMM 中分别负责什么。
