# 三阶段学习地图

这份地图说明三个项目为什么要分层，以及每一层学完以后应该具备什么能力。你可以把它当成课程大纲，也可以把它当成每次复盘时的验收清单。

## 总览

| 阶段 | 定位 | 主要算子 | 学习目标 | 毕业标准 |
| --- | --- | --- | --- | --- |
| Stage 01 Basic Kernels | 基础阶段 | `copy`、`vector_add`、`axpy`、`row_sum`、`row_sum_parallel` | 会读最小 TileLang kernel，理解 block/thread、全局下标和边界保护 | 能独立解释 `idx = bx * block_size + i`，并让整除和非整除 shape 都通过测试 |
| Stage 02 Advanced Ops | 进阶阶段 | `gemm`、`row_softmax`、`rmsnorm`、parallel reduction、`gelu`、`linear_bias_gelu` | 会解释 GEMM 数据流、reduction、数值稳定和 benchmark | 能画出 global/shared/fragment/global 的 GEMM 路径，并解释 fp16 误差和 `T.reduce_*` |
| Stage 03 Decoder Block Pipeline | 大型综合阶段 | RMSNorm、QKV GEMM、causal attention、optimized reductions、MLP、LM Head | 会把多个算子串成单 Decoder Block + logits 推理流水线 | 能从 `x=(B,S,H)` 一路追踪到 `logits=(B,S,V)`，并通过 reference 对齐 |

## 阶段 1：Basic Kernels

### 你要学会什么

- TileLang kernel 的最小结构：`@tilelang.jit`、`T.prim_func`、`T.Kernel`。
- GPU grid/block/thread 的粗略分工。
- 一维 elementwise 的通用写法。
- 为什么 `ceildiv` 会多启动一些 thread 位置。
- 为什么必须写 `if idx < N`。
- `row_sum_parallel_tilelang` 为什么要求一行放进一个 block。
- PyTorch reference 如何作为 correctness 标准答案。

### 必须完成的命令

```bash
python3 scripts/check_project.py --stage basic
pytest tests/test_basic.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -q -m tilelang
python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2
```

### 进入下一阶段的门槛

- 能解释 `N=1024` 和 `N=1000` 的区别。
- 能指出 `block_size` 变大或变小时 grid 数量如何变化。
- 能说明 `row_sum` 为什么不再是每个输出只依赖一个输入。
- 已完成 `reports/stage01_basic_template.md` 的学习记录。

## 阶段 2：Advanced Ops

### 你要学会什么

- GEMM 的 `M/N/K` 语义。
- tile、shared memory、fragment accumulator 在 GEMM 中分别做什么。
- `T.copy`、`T.clear`、`T.gemm`、`T.copy` 的执行顺序。
- softmax 为什么要做 max-subtraction。
- RMSNorm 为什么内部通常用 fp32 统计量。
- parallel reduction 如何用 `T.reduce_sum/T.reduce_max` 替换单线程串行循环。
- benchmark 结果应该如何记录，而不是只看一个数字。

### 必须完成的命令

```bash
python3 scripts/check_project.py --stage advanced
pytest tests/test_advanced.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -q -m tilelang
python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

### 进入下一阶段的门槛

- 能画出 `A/B global tile -> shared memory -> fragment -> C global`。
- 能解释为什么 v1 `gemm_tilelang` 只接受 tile-aligned shape。
- 能说出 stable softmax 的公式。
- 能解释串行 reduction 与并行 reduction 的数据流区别。
- 能根据 dtype 选择合理的 `atol/rtol`。
- 已完成 `reports/stage02_advanced_template.md` 的学习记录。

## 阶段 3：Decoder Block Pipeline

### 你要学会什么

- `MiniDecoderConfig` 中每个字段如何决定 shape。
- `hidden_size == num_heads * head_dim` 为什么必须成立。
- QKV linear 后如何 reshape/split head。
- causal attention 的 `QK^T -> softmax -> P@V`。
- 两次 RMSNorm、两次 residual add 和 MLP 的位置。
- optimized 入口只替换 RMSNorm 和 softmax reduction，不改变整体数学公式。
- logits 的 shape 为什么是 `(B, S, vocab_size)`。

### 必须完成的命令

```bash
python3 scripts/check_project.py --stage decoder
pytest tests/test_decoder.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_decoder.py -q -m tilelang
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --compare-optimized --warmup 1 --repeat 1
```

### 完成项目的门槛

- 能从 `x=(B,S,H)` 推导到 `q/k/v=(B,num_heads,S,head_dim)`。
- 能解释 causal mask 屏蔽未来 token，而不是过去 token。
- 能说明 residual 为什么需要保存原输入。
- 能让 `mini_inference_tilelang` 的 logits 与 PyTorch reference close。
- 已完成 `reports/stage03_decoder_template.md` 的学习记录。

## 推荐复盘方法

每完成一个阶段，都写下三件事：

1. 这层最重要的一个 TileLang 语义是什么。
2. 这层最容易出错的一个 shape 是什么。
3. 如果要把这层变快，下一步会优化哪里。

如果这三个问题答不出来，不急着往后走，回到对应的 `docs/kernels/*.md` 和测试文件再读一遍。
