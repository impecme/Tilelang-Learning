# 02 Advanced Ops

这一层进入 AI 算子中最常见的模式：GEMM、softmax、norm、activation 和 fusion。你会从“一个 thread 处理一个元素”的直觉，过渡到“一个 tile 如何被搬运、累加、写回”的直觉。

详细任务见 `../docs/stages/02_advanced_ops.md`，三阶段总览见 `../docs/stage_map.md`。

## 阶段定位

- 阶段：进阶阶段。
- 学习对象：矩阵乘法、reduction、数值稳定、融合接口。
- 核心代码：`../tilelab/advanced.py`。
- 测试文件：`../tests/test_advanced.py`。
- 报告模板：`../reports/stage02_advanced_template.md`。
- 动手 lab：`../labs/02_gemm_tile_shape/README.md`、`../labs/03_reduction_optimization/README.md`。

## 细化学习目标

- 会解释 GEMM 中 `M/N/K` 的含义。
- 会画出 `A/B global tile -> shared memory -> fragment -> C global`。
- 会说明 `T.copy`、`T.clear`、`T.gemm` 的角色。
- 会解释 stable softmax 为什么要减每行最大值。
- 会说明 RMSNorm 的平方均值和 fp32 统计量。
- 会对比串行 reduction 和并行 reduction 的运行方式。
- 会用 benchmark 记录性能观察，而不是只看一次运行时间。

## 算子清单

| 算子 | 公式 | 输入 | 输出 | 学习重点 |
| --- | --- | --- | --- | --- |
| `gemm` | `C = A @ B` | `A=(M,K), B=(K,N)` | `C=(M,N)` | tile、shared、fragment、`T.gemm` |
| `row_softmax` | `exp(x-m)/sum(exp(x-m))` | `x=(M,N)` | `y=(M,N)` | max-subtraction |
| `rmsnorm` | `x / rms(x) * weight` | `x=(M,N), w=(N,)` | `y=(M,N)` | reduction + scale |
| `row_softmax_parallel` | 同上 | `x=(M,N)` | `y=(M,N)` | `T.reduce_max/sum` |
| `rmsnorm_parallel` | 同上 | `x=(M,N), w=(N,)` | `y=(M,N)` | 单 block 并行 reduction |
| `gelu` | `0.5*x*(1+erf(x/sqrt(2)))` | `x` | `y` | elementwise 非线性 |
| `linear_bias_gelu` | `gelu(x@W+b)` | `x,W,b` | `y` | fusion 思路 |

## GEMM 数据流

GEMM 的 TileLang kernel 采用经典路径：

```text
A/B global tile -> shared memory -> fragment accumulator -> C global
```

代码里对应：

```python
A_shared = T.alloc_shared(...)
B_shared = T.alloc_shared(...)
C_local = T.alloc_fragment(...)
T.copy(A tile, A_shared)
T.copy(B tile, B_shared)
T.gemm(A_shared, B_shared, C_local)
T.copy(C_local, C tile)
```

`C_local` 要先 `T.clear`，因为 GEMM 是沿 K 维不断累加。

本项目 v1 的 GEMM 是教学版，不实现 tail tile。调用 `gemm_tilelang` 时必须满足：

```text
M % block_m == 0
N % block_n == 0
K % block_k == 0
```

如果不满足，会抛出明确 `ValueError`，避免 kernel silent 越界。

## 逐步任务

1. 读 `../docs/kernels/gemm.md`，画出 GEMM 的四段数据流。
2. 运行一次 tile 对齐 GEMM，再观察一次非 tile 对齐 shape 的报错。
3. 读 `row_softmax_tilelang`，找出 max、sum、normalize 三段。
4. 读 `rmsnorm_tilelang`，找出平方和、rsqrt、scale 三段。
5. 读 `../docs/reduction_optimization.md`，说明 parallel 版本为什么比串行版更接近高性能写法。
6. 读 `linear_bias_gelu_tilelang`，说明它如何体现 fusion 思路。
7. 读 `../docs/performance_guide.md`，理解 benchmark CSV、kernel launch 和 memory-bound。
8. 把 correctness、tolerance、benchmark 都写入进阶报告。

## 运行

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

## 性能学习入口

本阶段新增了几个小型教学优化 kernel：

- `scale_tilelang`
- `add_bias_tilelang`
- `linear_bias_tilelang`
- `scale_causal_mask_tilelang`

它们帮助你观察小操作的 kernel launch 和 memory traffic。详细思路见 `../docs/performance_guide.md` 和 `../docs/optimization_log.md`。

本阶段也提供并行 reduction 对照：

- `row_softmax_parallel_tilelang`
- `rmsnorm_parallel_tilelang`

详细思路见 `../docs/reduction_optimization.md`。

## 验收标准

- 能说出 `M/N/K` 分别对应什么。
- 能解释 shared memory 和 fragment 在 GEMM 中的角色。
- 能手写 stable softmax 公式。
- 能说明 RMSNorm 和 LayerNorm 的核心区别。
- 能用 tolerance 解释 fp16/bf16 correctness。
- 能说明教学版串行 reduction 为什么易懂但不高性能。
- 能解释 `T.reduce_sum/T.reduce_max` 在 parallel reduction 中的作用。
- 能运行 gemm/reduction lab，并解释一个 shape guard。
- 能读懂 benchmark CSV，并解释小操作为什么常常是 memory-bound。

## 常见错误

- 用非 tile 对齐 shape 调用 `gemm_tilelang`。
- 忘记 `T.clear` 导致 fragment 累加器保留旧值。
- 直接 `exp(x)` 导致 softmax overflow。
- 只比较 benchmark，不先确认 correctness。
- 把本项目 v1 当成最终性能实现。

## 进入下一阶段条件

完成 `../reports/stage02_advanced_template.md` 的学习记录，并且能独立解释 GEMM 数据流和 softmax/RMSNorm reduction 后，再进入 `03_decoder_block_pipeline`。
