# Stage 04 - GEMM 主线

## 阶段目标

这一阶段，我以 GEMM 为主线，把 TileLang 的 tile、shared memory、fragment、Tensor Core、pipeline 串起来。GEMM 是 AI 算子开发训练场，也是后续 attention 中 `QK^T` 和 `P@V` 的基础。

## 先修状态

- 已完成 Stage 03。
- 能跑 benchmark，并能解释 latency 与 warmup/repeat。
- 理解 global/shared/fragment 的基本区别。

## 阅读

- TileLang GEMM examples。
- `notes/concepts_deep_dive.md` 第 5、6、9、10 节。
- TileLang Instructions。
- TileLang Software Pipeline Annotations。
- `kernels/gemm.py`。
- `benchmarks/bench_gemm.py`。
- `tests/test_gemm.py`。

## 概念

- GEMM 数学：`C[M, N] = A[M, K] @ B[K, N]`。
- `M/N/K`：输出行、输出列、reduce 维。
- `block_M/block_N`：一个 block 负责的输出 tile。
- `block_K`：每次沿 K 维加载和累加的 tile 深度。
- shared staging：把 global memory tile 先搬进 shared memory 复用。
- fragment accumulator：把局部输出 tile 存在 fragment/register 中累加。
- fp32 accumulation：输入 fp16/bf16 时用 fp32 累加降低误差。
- Tensor Core/MMA：矩阵乘硬件指令路径。
- warp partition：warp 如何分摊 tile 计算。
- software pipeline：copy 和 compute 分阶段重叠。
- `T.gemm`：TileLang 的 GEMM intrinsic。
- `T.Pipelined`：表达 pipeline loop。
- `T.clear`：初始化 accumulator。
- `T.copy`：tile 搬运。

## 代码

- `kernels/gemm.py`
  - `_compile_gemm` 参数。
  - `A_shared`、`B_shared`、`C_local`。
  - `T.Pipelined(T.ceildiv(K, block_K), num_stages=...)`。
  - `T.gemm(A_shared, B_shared, C_local)`。
  - `T.copy(C_local, C[...])`。
- `tests/test_gemm.py`
  - `128x128x32` baseline。
- `benchmarks/bench_gemm.py`
  - `--run-tilelang`。

## 练习

1. 跑 baseline：

   ```bash
   RUN_TILELANG_SMOKE=1 pytest tests/test_gemm.py -m tilelang
   python3 -m benchmarks.bench_gemm --m 128 --n 128 --k 128 --warmup 10 --repeat 50 --run-tilelang
   ```

2. 扫描少量配置：
   - 固定 baseline：`block_m=128, block_n=128, block_k=32`。
   - 尝试不同 `threads`。
   - 尝试不同 `num_stages`。
   - 不要一次改变太多参数。

3. 记录性能：
   - latency ms。
   - TFLOPS：`2 * M * N * K / latency_seconds / 1e12`。
   - PyTorch matmul latency。
   - TileLang latency。
   - correctness 是否通过。

4. 数据流说明：
   - 写一段文字说明 A/B/C 分别在哪里。
   - 说明每次 K loop 搬运什么。
   - 说明为什么 C_local 要先 `T.clear`。

## 报告模板

建议写 `reports/stage04_gemm.md`：

```markdown
# Stage 04 GEMM Report

## Baseline Config

## Correctness

## Benchmark Table

| M | N | K | block_M | block_N | block_K | threads | stages | torch ms | tilelang ms | TFLOPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Data Flow

## Bottleneck Guess

## Next Experiments
```

## 思考问题

- 为什么 GEMM 的计算量是 `2*M*N*K`？
- 为什么 `block_K` 太大或太小都可能不好？
- shared memory staging 的复用体现在哪里？
- GEMM 和 attention 中 `QK^T`、`P@V` 的关系是什么？

## 验收标准

- GEMM correctness 通过。
- 至少有 3 组配置 benchmark。
- 能解释一次 GEMM 中数据如何从 global 到 shared 到 fragment 再写回 global。
- 能说明为什么 GEMM 是 attention 的基础。
