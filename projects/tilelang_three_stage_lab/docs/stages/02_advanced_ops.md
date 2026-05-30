# Stage 02：Advanced Ops

进阶阶段开始接近真实 AI kernel。你会看到 GEMM、softmax、RMSNorm、GELU 和 fusion。学习重点从“一个 thread 处理一个元素”扩展到“一个 tile 如何被搬运、累加和写回”。

## 阶段定位

- 难度：进阶。
- 核心文件：`tilelab/advanced.py`。
- 对应测试：`tests/test_advanced.py`。
- 对应 benchmark：`benchmarks/bench_advanced.py`。
- 对应 lab：`labs/02_gemm_tile_shape/README.md`、`labs/03_reduction_optimization/README.md`。
- 学习报告：`reports/stage02_advanced_template.md`。

## 细化学习目标

- 能从 `A=(M,K), B=(K,N), C=(M,N)` 推导 GEMM 的三个维度。
- 能解释 shared memory 为什么比每次都读 global memory 更接近高性能写法。
- 能说明 fragment accumulator 为什么需要 `T.clear`。
- 能读懂 `T.copy -> T.gemm -> T.copy` 的基本数据流。
- 能写出 stable softmax 公式。
- 能解释 RMSNorm 的平方均值、`eps` 和 fp32 accumulation。
- 能对比串行 reduction 和并行 reduction 的数据流。
- 能说明 `linear_bias_gelu` 为什么体现了 fusion 思路。
- 能用 `bench_optimization` 观察 scale、bias add、mask 这类小操作的开销。
- 能把 benchmark 结果写成 CSV，并给每一行补充解释。

## 必会概念

| 概念 | 小白解释 |
| --- | --- |
| tile | 把大矩阵切成小块来搬运和计算 |
| shared memory | block 内共享的高速临时存储 |
| fragment | 用来保存局部累加结果的小块寄存器/片段 |
| `T.copy` | 在 global/shared/fragment 等存储之间搬数据 |
| `T.gemm` | 调用 TileLang 的矩阵乘累加语义 |
| stable softmax | 先减最大值再指数化，避免溢出 |
| parallel reduction | 一个 block 的多个线程共同聚合一行数据 |
| `T.reduce_sum/max` | TileLang 中表达 block 内 reduction 的教学入口 |
| fusion | 把多个操作组合，减少中间 tensor 读写 |
| kernel launch | 启动一个 GPU kernel 的调度开销 |
| CSV benchmark | 用结构化表格记录 shape、backend、latency 和备注 |

## 逐步任务

1. 阅读 `docs/kernels/gemm.md`，画出 `A/B global tile -> shared memory -> fragment -> C global`。
2. 在 `gemm_tilelang` 调用处观察 `block_m/block_n/block_k`，解释为什么 v1 要求 shape 对齐。
3. 阅读 `row_softmax_tilelang`，找出 max、sum、normalize 三段。
4. 阅读 `rmsnorm_tilelang`，找出平方和、rsqrt、scale 三段。
5. 阅读 `docs/reduction_optimization.md`，对比 serial 和 parallel reduction。
6. 运行 `python3 scripts/run_lab.py --lab gemm` 和 `python3 scripts/run_lab.py --lab reduction`。
7. 阅读 `gelu_tilelang` 和 `linear_bias_gelu_tilelang`，说明 elementwise activation 与 GEMM 串联的关系。
8. 阅读 `scale_tilelang`、`add_bias_tilelang`、`scale_causal_mask_tilelang`，理解为什么小操作也值得单独观察。
9. 运行测试和 benchmark，把 correctness、tolerance、latency 和 CSV 路径都写进报告。

## 必须运行

```bash
python3 scripts/check_project.py --stage advanced
python3 scripts/run_lab.py --lab gemm
python3 scripts/run_lab.py --lab reduction
pytest tests/test_advanced.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -q -m tilelang
python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2
python3 -m benchmarks.bench_optimization --run-tilelang --warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

## 实验记录要求

- 记录 GEMM 的 `M/N/K/block_m/block_n/block_k`。
- 记录 softmax 输入含大数时是否仍然正确。
- 记录 RMSNorm 和 GELU 的 dtype 与 tolerance。
- 至少写一次 benchmark 观察：哪个算子更像 memory-bound，哪个更像 compute-bound。
- 记录 `bench_optimization` 生成的 CSV，并解释 `scale/add_bias/scale_causal_mask` 的 latency 差异。
- 记录 `bench_reductions` 生成的 CSV，并解释 serial 与 parallel reduction 的差异。

## 验收问题

- `M/N/K` 分别来自哪个矩阵的哪个维度？
- `T.copy` 和 `T.gemm` 分别负责什么？
- 为什么 fragment 要先清零？
- softmax 为什么不是直接 `exp(x) / sum(exp(x))`？
- 为什么这个项目的串行 reduction 易懂但不高性能？
- `T.reduce_max` 和 `T.reduce_sum` 在 parallel softmax 中分别负责什么？
- 为什么 parallel v1 要求 `cols <= block_n`？
- 为什么 `linear_bias_tilelang` 还不算真正的 GEMM epilogue fusion？
- 什么时候一个小操作的 kernel launch 开销会变得明显？

## 常见卡点

- 用非 tile 对齐 shape 调用 `gemm_tilelang`。
- 把 `A=(M,K)` 和 `B=(K,N)` 的 K 维对不上。
- 忘记 fp16 的误差比 fp32 大。
- 只比较 latency，不先确认 correctness。

## 进入下一阶段条件

当你能独立解释 GEMM 的数据流，并能说明 softmax/RMSNorm 的 reduction 过程时，再进入 Stage 03。
