# Performance Guide

这份文档不是教你一次性写出最快 kernel，而是给你一个可重复的性能学习框架：先保证正确，再定位瓶颈，只改一个变量，最后把数据记录下来。

## 快速上手

第一轮只跑三个命令：

```bash
python3 -m benchmarks.bench_optimization --run-tilelang --warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_advanced.csv
python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv
```

然后打开 CSV，看每一行的 `suite/name/backend/shape/latency_ms/notes`。

也可以用脚本先做一次自动解读：

```bash
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_decoder.csv
```

这个脚本会按 `suite/shape/dtype` 分组，打印 fastest/slowest、torch vs TileLang、serial vs parallel、baseline vs optimized。它不是替你下最终结论，而是帮你先把数据看清楚。

## 性能学习顺序

1. correctness：先用 `torch.testing.assert_close` 确认结果。
2. warmup：第一次 JIT 编译和 CUDA 初始化不算稳定 latency。
3. baseline：先记录 PyTorch reference，再记录 TileLang。
4. single variable：一次只改一个变量，例如 shape、block size、dtype 或 kernel 组合。
5. interpretation：写下为什么快或慢，不只保存数字。

## 为什么要优化小操作

Decoder TileLang 路径里原来有一些 PyTorch 小操作：

```text
out + bias
scores * scale
scores masked_fill causal mask
```

这些操作本身很小，但会引入额外 kernel launch 或框架调度。v1 优化把它们换成教学版 TileLang kernels：

- `scale_tilelang`
- `add_bias_tilelang`
- `linear_bias_tilelang`
- `scale_causal_mask_tilelang`

它们不是最终高性能 epilogue fusion，但能帮助你理解“减少中间操作”和“把数据流留在 TileLang 路径里”的思路。

## 为什么要做并行 Reduction

`row_sum`、`row_softmax`、`rmsnorm` 都有“先看完整一行，再输出结果”的数据依赖。旧版教学 kernel 使用 `threads=1`，好读但一行只有一个线程在工作。

新增 parallel 版本使用：

- `row_sum_parallel_tilelang`
- `row_softmax_parallel_tilelang`
- `rmsnorm_parallel_tilelang`

它们让一个 CUDA block 的多个线程共同处理一行，再用 `T.reduce_sum` 或 `T.reduce_max` 汇总。详细数据流见 `docs/reduction_optimization.md`。

## Memory-Bound 和 Compute-Bound

简单判断：

- elementwise、bias add、scale 常常 memory-bound，因为每个元素计算很少，主要在搬数据。
- GEMM 常常 compute-bound，因为乘加次数多，Tensor Core 利用率很关键。
- softmax/RMSNorm 既有 reduction 又有读写，教学版串行 reduction 主要是学习语义，不代表性能上限。

不要只凭感觉判断。至少记录 shape、dtype、latency 和备注。

## CSV 字段

| 字段 | 含义 |
| --- | --- |
| `suite` | benchmark 分组，例如 `advanced`、`decoder_block`、`optimization` |
| `name` | 具体测试项 |
| `backend` | `torch` 或 `tilelang` |
| `shape` | 本次输入 shape |
| `dtype` | 本次 dtype |
| `latency_ms` | 平均耗时 |
| `warmup` | 预热次数 |
| `repeat` | 计时重复次数 |
| `notes` | 解释或限制 |

## 复盘模板

每次 benchmark 后回答：

- 哪个操作最慢？
- 它更像 memory-bound 还是 compute-bound？
- TileLang 版本是否更快？如果没有，可能原因是什么？
- `summarize_bench_csv.py` 给出的 speedup 是否符合你的直觉？如果不符合，先检查 shape 和 notes。
- 下一轮只改一个什么变量？
- 这个优化是否改变了数学结果？

## 当前 v1 限制

- 不做 autotune。
- 并行 reduction v1 只支持 `cols <= block_n <= 1024` 的单 block 教学 shape。
- `linear_bias_tilelang` 是 `gemm_tilelang + add_bias_tilelang`，不是真正 GEMM epilogue fusion。
- benchmark 数字用于学习和对比，不代表最终性能结论。
