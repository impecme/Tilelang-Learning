# TileLang Three Stage Lab

这是一个三层 TileLang 学习项目，目标是让刚入门的人从最小 kernel 一路走到一个迷你 Decoder Block 推理流水线。

如果你还没有跑通过任何 TileLang kernel，建议先完成轻量入门版 `../tilelang_starter_three_stage_lab/README.md`。starter 更少、更浅，适合第一轮建立直觉；本项目是第二轮系统学习版，文档、测试和工程检查更严格。

本项目不追求第一版性能超过 PyTorch。它优先保证：

- 每个算子都有 PyTorch reference。
- 每个 TileLang kernel 都有 correctness test。
- 每一层都有中文说明：功能、结构、张量语义、TileLang 语义、常见错误和验收标准。
- benchmark 是学习工具，用来观察 latency 和瓶颈，不是比赛成绩。

## 快速开始

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_three_stage_lab
python3 -m pip install -r requirements.txt
python3 scripts/check_project.py
pytest
python3 scripts/run_lab.py --lab all
```

可选 TileLang 编译 smoke：

```bash
RUN_TILELANG_SMOKE=1 pytest -m tilelang
```

benchmark smoke：

```bash
python3 -m benchmarks.bench_basic
python3 -m benchmarks.bench_advanced
python3 -m benchmarks.bench_reductions
python3 -m benchmarks.bench_decoder_block
```

真正运行 TileLang benchmark：

```bash
python3 -m benchmarks.bench_basic --run-tilelang
python3 -m benchmarks.bench_advanced --run-tilelang
python3 -m benchmarks.bench_optimization --run-tilelang --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --csv /tmp/tilelang_reductions.csv
python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```

动手 Lab：

```bash
python3 scripts/run_lab.py --list
python3 scripts/run_lab.py --lab all
python3 scripts/demo_common_errors.py --case all
```

## 目录结构

```text
tilelang_three_stage_lab/
  01_basic_kernels/            # 基础项目说明
  02_advanced_ops/             # 进阶项目说明
  03_decoder_block_pipeline/   # 大型项目说明
  labs/                        # 动手 Lab：读代码、跑检查、改参数、自测
  docs/                        # 阶段地图、术语、shape、debug、逐 kernel 教程、benchmark 模板
  tilelab/                     # Python package：reference 和 TileLang 实现
  tests/                       # correctness tests
  benchmarks/                  # latency smoke
  reports/                     # 学习报告和 benchmark 输出
  scripts/                     # 项目健康检查脚本
```

代码放在 `tilelab/`，因为 Python package 名不能以数字开头。三个数字目录保留给学习文档，让阅读顺序一眼能看懂。

## 从零开始怎么学

如果你第一次打开这个项目，按这个顺序来：

1. 运行 `python3 scripts/check_project.py`，确认环境和入口文件齐全。
2. 阅读 `docs/stage_map.md`，先理解三阶段的难度递进和毕业标准。
3. 阅读 `docs/lab_index.md`，先做一个可运行 Lab。
4. 阅读 `docs/learning_path.md`，按 Day 1 到 Day 7 推进。
5. 每学一个阶段，先跑 `python3 scripts/check_project.py --stage <basic|advanced|decoder>`。
6. 每学一个算子，先读 `docs/kernels/<op>.md`，再读 `tilelab/*.py` 中对应的 `xxx_reference` 和 `xxx_tilelang`。
7. 跑对应测试，例如 `pytest tests/test_basic.py`。
8. 如果是 TileLang kernel，跑 `RUN_TILELANG_SMOKE=1 pytest -m tilelang`。
9. 把实验记录写进 `reports/stage*_template.md` 的副本。
10. 做性能学习时，先读 `docs/performance_guide.md`，再跑 benchmark，最后用 `scripts/summarize_bench_csv.py` 辅助解读 CSV。

## 性能学习入口

本项目的性能目标是“会观察、会解释、会记录”，不是第一版就超过 PyTorch。推荐命令：

```bash
python3 -m benchmarks.bench_optimization --run-tilelang --warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_advanced.csv
python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_decoder.csv
```

阅读顺序：

1. `docs/performance_guide.md`
2. `docs/optimization_log.md`
3. `docs/reduction_optimization.md`
4. `docs/benchmark_template.md`
5. `docs/lab_answer_key.md`
6. `reports/perf_optimization_template.csv`
7. `reports/reduction_optimization_template.md`

## 阶段化学习地图

项目有三层，但它们不是三个平级 demo，而是一条递进路线：

| 阶段 | 你在学什么 | 你要交付什么 | 进入下一阶段条件 |
| --- | --- | --- | --- |
| `basic` | 最小 kernel、线程映射、边界保护、输入检查 | 基础报告、整除/非整除测试记录 | 能解释 `idx` 和 boundary guard |
| `advanced` | GEMM、shared memory、fragment、reduction、数值稳定 | 进阶报告、GEMM/softmax/norm benchmark 记录 | 能画出 GEMM 数据流并解释误差 |
| `decoder` | 单 Decoder Block + logits 的完整 shape 流水线 | 综合报告、`x -> logits` shape 记录 | 能对齐 TileLang logits 与 PyTorch reference |

机器可读的阶段定义在 `tilelab/stages.py`，脚本和测试会读取它，避免 README 写一套、检查脚本跑另一套。

阶段检查命令：

```bash
python3 scripts/check_project.py --list-stages
python3 scripts/check_project.py --stage basic
python3 scripts/check_project.py --stage advanced
python3 scripts/check_project.py --stage decoder
```

## 三层学习路线

1. `01_basic_kernels`
   - 算子：`vector_add`、`copy`、`axpy`、`row_sum`。
   - 目标：理解一个元素由哪个 block/thread 处理，为什么需要 boundary guard。
   - 学习重点：会读、会跑、会改边界。
   - 详细阶段说明：`docs/stages/01_basic_kernels.md`。

2. `02_advanced_ops`
   - 算子：`gemm`、`row_softmax`、`rmsnorm`、`row_softmax_parallel`、`rmsnorm_parallel`、`gelu`、`linear_bias_gelu`。
   - 目标：理解 shared memory、fragment、reduction、数值稳定和 fusion。
   - 学习重点：会解释数据搬运、串行/并行 reduction、误差和 benchmark。
   - 详细阶段说明：`docs/stages/02_advanced_ops.md`。

3. `03_decoder_block_pipeline`
   - 流水线：

     ```text
     x -> RMSNorm -> QKV Linear -> Causal Attention
       -> Output Linear -> Residual -> RMSNorm
       -> MLP(GELU) -> Residual -> LM Head
     ```

   - 目标：把多个 TileLang 算子串成一个可测试的迷你推理模块。
   - 学习重点：会跟踪 shape、权重、残差、attention 和 logits 数据流。
   - 详细阶段说明：`docs/stages/03_decoder_block_pipeline.md`。

## 公共接口约定

所有算子都采用成对命名：

- `xxx_reference(...)`：PyTorch 参考实现，用来给 correctness test 提供标准答案。
- `xxx_tilelang(...)`：TileLang 实现或 TileLang 编排实现。

大型项目提供：

- `MiniDecoderConfig`
- `MiniDecoderWeights`
- `make_random_weights(config, seed, device)`
- `decoder_block_reference(x, weights, config)`
- `decoder_block_tilelang(x, weights, config)`
- `decoder_block_tilelang_optimized(x, weights, config)`
- `mini_inference_reference(x, weights, config)`
- `mini_inference_tilelang(x, weights, config)`
- `mini_inference_tilelang_optimized(x, weights, config)`

## TileLang v1 限制

- GEMM 教学 kernel 使用固定 tile 思路，要求 `M/N/K` 分别能被 `block_m/block_n/block_k` 整除；不满足会抛出明确 `ValueError`。
- `linear_bias_tilelang` 是教学组合版 `gemm_tilelang + add_bias_tilelang`，不是真正 GEMM epilogue fusion。
- `scale_causal_mask_tilelang` 是小型 fusion 示例，用来学习减少中间操作。
- `row_sum`、`row_softmax`、`rmsnorm` 是 correctness-first kernel，使用单线程串行 reduction，方便理解，不代表高性能写法。
- `row_sum_parallel_tilelang`、`row_softmax_parallel_tilelang`、`rmsnorm_parallel_tilelang` 是单 block 并行 reduction 教学版，只支持 `cols <= block_n <= 1024`。
- `decoder_block_tilelang_optimized` 只替换 RMSNorm 和 attention softmax 的 reduction 路径，不是完整 fused attention。
- Decoder Block 的 attention 用 Python 循环编排多个 TileLang 小算子，重点是学习数据流，不是最终优化形态。
- 不做 tokenizer、采样、训练和 KV cache。

## 推荐阅读顺序

1. `docs/learning_path.md`
2. `docs/lab_index.md`
3. `docs/stage_map.md`
4. `docs/glossary.md`
5. `docs/tilelang_semantics.md`
6. `01_basic_kernels/README.md`
7. `docs/stages/01_basic_kernels.md`
8. `docs/kernels/vector_add.md`
9. `02_advanced_ops/README.md`
10. `docs/stages/02_advanced_ops.md`
11. `docs/kernels/gemm.md`
12. `docs/tensor_shapes.md`
13. `docs/shape_walkthroughs.md`
14. `03_decoder_block_pipeline/README.md`
15. `docs/stages/03_decoder_block_pipeline.md`
16. `docs/kernels/decoder_block.md`
17. `docs/debug_checklist.md`
18. `docs/error_gallery.md`
19. `docs/benchmark_template.md`
20. `docs/performance_guide.md`
21. `docs/optimization_log.md`
22. `docs/reduction_optimization.md`

## 逐 Kernel 教程索引

- `docs/kernels/vector_add.md`
- `docs/kernels/copy.md`
- `docs/kernels/axpy.md`
- `docs/kernels/row_sum.md`
- `docs/kernels/gemm.md`
- `docs/kernels/softmax.md`
- `docs/kernels/rmsnorm.md`
- `docs/kernels/gelu.md`
- `docs/kernels/decoder_block.md`
- `docs/reduction_optimization.md`
