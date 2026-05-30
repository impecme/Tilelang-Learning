# Learning Path

这是一条 7 天学习路线。每天都按同一个节奏走：先读文档，再读代码，再跑测试，最后回答验收问题。

三阶段关系：

- Day 1-3：Stage 01 Basic Kernels，建立最小 TileLang kernel 直觉。
- Day 4-5：Stage 02 Advanced Ops，学习 GEMM、reduction、数值稳定和 benchmark。
- Day 6-7：Stage 03 Decoder Block Pipeline，把算子串成迷你推理流水线并复盘。
- 性能学习穿插在 Day 5-7：先读性能框架，再跑 CSV benchmark，最后解释结果。

每个阶段都有一个明确的“不要急着往后走”门槛。门槛写在 `docs/stage_map.md` 和 `docs/stages/*.md` 中，也可以用 `python3 scripts/check_project.py --stage <id>` 查看。

## Day 1 - 项目启动

阅读：

- `README.md`
- `docs/stage_map.md`
- `docs/lab_index.md`
- `docs/glossary.md`
- `docs/tilelang_semantics.md`

命令：

```bash
python3 scripts/check_project.py
python3 scripts/check_project.py --list-stages
python3 scripts/run_lab.py --list
pytest -q
```

验收问题：

- `xxx_reference` 和 `xxx_tilelang` 分别负责什么？
- lab、test、benchmark 分别解决什么问题？
- 为什么 TileLang kernel 通常要求 CUDA tensor？
- 为什么第一次 JIT 编译耗时不适合作为 benchmark？

产出：

- 画出三阶段学习顺序。
- 选定本周要填写的三份报告模板。

## Day 2 - 一维 Elementwise

阅读：

- `01_basic_kernels/README.md`
- `docs/stages/01_basic_kernels.md`
- `docs/kernels/vector_add.md`
- `labs/01_basic_kernel_debug/README.md`
- `docs/kernels/copy.md`
- `docs/kernels/axpy.md`

命令：

```bash
python3 scripts/check_project.py --stage basic
python3 scripts/run_lab.py --lab basic
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -m tilelang
python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2
```

验收问题：

- `idx = bx * block_size + i` 是什么？
- `N=1000` 时为什么必须有 `if idx < N`？
- `alpha` 在 AXPY 里为什么适合作为编译期参数？

产出：

- 在 `reports/stage01_basic_template.md` 的副本中记录 `N=1024` 和 `N=1000`。

## Day 3 - Reduction 入门

阅读：

- `docs/kernels/row_sum.md`
- `docs/reduction_optimization.md`
- `docs/error_gallery.md`
- `docs/debug_checklist.md`

命令：

```bash
python3 scripts/check_project.py --stage basic
pytest tests/test_basic.py
python3 scripts/demo_common_errors.py --case cpu_tensor
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

验收问题：

- elementwise 和 reduction 的核心区别是什么？
- 为什么本项目的 `row_sum_tilelang` 易懂但不高性能？
- `row_sum_parallel_tilelang` 为什么要求 `cols <= block_n`？

Stage 01 门槛：

- 能独立解释 boundary guard。
- 能根据报错判断 shape、dtype、device、contiguous 哪一类出了问题。
- 已完成基础阶段学习报告。

## Day 4 - GEMM

阅读：

- `02_advanced_ops/README.md`
- `docs/stages/02_advanced_ops.md`
- `docs/kernels/gemm.md`
- `labs/02_gemm_tile_shape/README.md`

命令：

```bash
python3 scripts/check_project.py --stage advanced
python3 scripts/run_lab.py --lab gemm
RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -m tilelang
python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2
```

验收问题：

- `M/N/K` 分别表示什么？
- `T.copy` 和 `T.gemm` 各自做什么？
- 为什么 v1 GEMM 要求 tile-aligned shape？

产出：

- 画出 GEMM 数据流。
- 在报告中记录一次 tile-aligned shape 和一次非对齐 shape 报错。

## Day 5 - Softmax、RMSNorm、GELU

阅读：

- `docs/kernels/softmax.md`
- `docs/kernels/rmsnorm.md`
- `docs/kernels/gelu.md`
- `docs/performance_guide.md`
- `docs/reduction_optimization.md`
- `labs/03_reduction_optimization/README.md`

命令：

```bash
python3 scripts/check_project.py --stage advanced
python3 scripts/run_lab.py --lab reduction
pytest tests/test_advanced.py
python3 -m benchmarks.bench_optimization --run-tilelang --warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

验收问题：

- softmax 为什么要减 max？
- RMSNorm 为什么内部用 fp32 统计量？
- GELU 和 ReLU 的差别在哪里？
- `scale/add_bias/scale_causal_mask` 更像 memory-bound 还是 compute-bound？
- 串行 reduction 和并行 reduction 的 latency 差异说明了什么？

Stage 02 门槛：

- 能解释 GEMM 的 global/shared/fragment/global 路径。
- 能说明 stable softmax 和 RMSNorm 的 reduction 过程。
- 能读懂 benchmark CSV 中的 `suite/name/backend/shape/latency_ms`。
- 能解释 `T.reduce_sum/T.reduce_max` 在 parallel reduction 中的作用。
- 已完成进阶阶段学习报告。

## Day 6 - Decoder Block

阅读：

- `03_decoder_block_pipeline/README.md`
- `docs/stages/03_decoder_block_pipeline.md`
- `docs/kernels/decoder_block.md`
- `docs/tensor_shapes.md`
- `docs/shape_walkthroughs.md`
- `labs/04_decoder_shape_trace/README.md`
- `docs/optimization_log.md`

命令：

```bash
python3 scripts/check_project.py --stage decoder
python3 scripts/run_lab.py --lab decoder
RUN_TILELANG_SMOKE=1 pytest tests/test_decoder.py -m tilelang
python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv
```

验收问题：

- `QK^T -> softmax -> P@V` 的 shape 分别是什么？
- causal mask 屏蔽的是未来还是过去？
- residual add 为什么要保留原输入？
- `decoder_block_tilelang_optimized` 相比旧入口替换了哪些 reduction 算子？

产出：

- 写出默认 config 下 `x -> logits` 的每一步 shape。
- 记录一次 `mini_inference_tilelang` 与 reference close 的结果。
- 记录一次 decoder benchmark CSV，并说明优化前后的编排差异。
- 对比 `tilelang_decoder_block` 和 `tilelang_decoder_block_optimized_reductions`。

## Day 7 - 复盘与报告

阅读：

- `docs/benchmark_template.md`
- `docs/performance_guide.md`
- `docs/optimization_log.md`
- `reports/stage01_basic_template.md`
- `reports/stage02_advanced_template.md`
- `reports/stage03_decoder_template.md`
- `reports/perf_optimization_template.csv`
- `reports/reduction_optimization_template.md`

命令：

```bash
python3 -m benchmarks.bench_basic --run-tilelang
python3 -m benchmarks.bench_advanced --run-tilelang --csv /tmp/tilelang_advanced.csv
python3 -m benchmarks.bench_optimization --run-tilelang --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --csv /tmp/tilelang_reductions.csv
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --compare-optimized --csv /tmp/tilelang_decoder.csv
```

验收问题：

- 哪个算子是 memory-bound？
- 哪个算子是 compute-bound？
- 下一步要优化哪个 kernel，为什么？
- CSV 的 `notes` 字段帮你记录了什么限制或解释？

Stage 03 门槛：

- 能从 `x=(B,S,H)` 推导到 `logits=(B,S,V)`。
- 能指出每个 TileLang kernel 在 Decoder Block 中的位置。
- 已完成综合阶段学习报告。
