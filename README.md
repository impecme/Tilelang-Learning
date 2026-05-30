# Tilelang-Learning

这是一个面向自己的 TileLang 学习工程。我会从 TileLang 入门开始，逐步补齐 Python/PyTorch、Transformer、GPU 算子和 AI 算子开发能力。目标不是只读文档，而是在每个学习阶段留下可运行代码、correctness test、benchmark 记录和复盘报告，最终完成一个 FlashAttention forward 原型。

## 环境基线

- 目标机器：NVIDIA A100，CUDA/PyTorch 环境优先。
- 工程固定版本：`tilelang==0.1.9`。
- 官网文档可能已经更新到更高版本；学习概念时参考官网，写代码时以本工程测试结果为基准。
- PyTorch 环境：使用当前系统环境中已经安装的 CUDA 版 PyTorch。
- 系统 `nvcc` 与 PyTorch CUDA 版本可能不同，所以第一阶段优先用 PyPI wheel，不做源码构建。

安装与检查：

```bash
cd /home/vipuser/Tilelang-Learning
python3 -m pip install -r requirements.txt
python3 scripts/check_env.py
python3 scripts/check_learning_project.py
pytest
```

可选 TileLang 编译 smoke test：

```bash
RUN_TILELANG_SMOKE=1 pytest -m tilelang
```

说明：`tilelang==0.1.9` 在当前环境导入时可能打印 TVM registry warning；目前不影响 import、测试或 kernel 编译。

## 工程结构

```text
Tilelang-Learning/
  README.md
  requirements.txt
  pyproject.toml
  kernels/        # TileLang 算子、PyTorch reference、毕业接口
  tests/          # correctness tests
  benchmarks/     # latency/TFLOPS benchmark 入口
  notes/          # 阶段式学习笔记、概念导读、算子 checklist
  reports/        # 阶段报告、benchmark 输出、最终复盘
  projects/       # 独立实战学习项目
  scripts/        # 环境检查和 smoke test
```

## 阶段式主路线

主路线不按周推进，而按掌握程度推进。每个阶段都围绕五件事展开：读什么、掌握哪些概念、看哪些代码、做哪些练习、如何验收。

| Stage | 主题 | 主要产物 |
| --- | --- | --- |
| 00 | 环境、工程结构、学习方法 | 环境报告、测试闭环、版本基线 |
| 00.25 | 面向 C++ 程序员的 Python/PyTorch 最小基础 | Python 语法迁移、PyTorch tensor、pytest、代码阅读能力 |
| 00.5 | 深度学习与 Transformer 最小背景 | Q/K/V、矩阵 shape、attention 手算与 PyTorch 练习 |
| 01 | GPU 算子基本心智模型 | 术语表、vector add 数据流、axpy 练习 |
| 02 | TileLang DSL 与基础控制流 | copy、row-wise sum/max、错误案例 |
| 03 | 内存层级、调试与性能观察 | reduction benchmark、debug 复盘 |
| 04 | GEMM 主线 | GEMM benchmark、tile 配置记录、TFLOPS |
| 05 | Autotuning 与常见 AI 算子模式 | autotune 记录、融合算子、softmax/RMSNorm |
| 06 | Attention 基线与 Online Softmax | online softmax 推导、attention baseline benchmark |
| 07 | FlashAttention Forward Capstone | non-causal FlashAttention forward、最终报告 |

阶段入口：

- [Learning Map - 从根笔记到 starter/full lab 的完整路线](notes/learning_map.md)
- [Stage 00 - 环境、工程结构、学习方法](notes/stage00_env.md)
- [Stage 00.25 - 面向 C++ 程序员的 Python/PyTorch 最小基础](notes/stage00_25_python_for_cpp.md)
- [Stage 00.5 - 深度学习与 Transformer 最小背景](notes/stage00_5_transformer_basics.md)
- [Stage 01 - GPU 算子基本心智模型](notes/stage01_gpu_operator_model.md)
- [Stage 02 - TileLang DSL 与基础控制流](notes/stage02_tilelang_dsl.md)
- [Stage 03 - 内存层级、调试与性能观察](notes/stage03_memory_debug_perf.md)
- [Stage 04 - GEMM 主线](notes/stage04_gemm.md)
- [Stage 05 - Autotuning 与常见 AI 算子模式](notes/stage05_autotune_operator_patterns.md)
- [Stage 06 - Attention 基线与 Online Softmax](notes/stage06_attention_baseline.md)
- [Stage 07 - FlashAttention Forward Capstone](notes/stage07_flash_attention.md)

概念入口：

- [Concepts - TileLang AI 算子开发概念导读](notes/concepts.md)
- [Concepts Deep Dive - TileLang AI 算子开发细讲版](notes/concepts_deep_dive.md)
- [Glossary - AI 算子开发中英文术语对照](notes/glossary_zh_en.md)
- [TileLang AI 算子开发 Checklist](notes/operator_checklist.md)

实战项目入口：

- [TileLang Starter Three Stage Lab - 轻量入门三层项目](projects/tilelang_starter_three_stage_lab/README.md)
- [TileLang Three Stage Lab - 基础、进阶、迷你 Decoder Block 三层项目](projects/tilelang_three_stage_lab/README.md)

推荐顺序：先跑 `tilelang_starter_three_stage_lab` 建立第一轮直觉，再进入 `tilelang_three_stage_lab` 做系统学习和严格验收。

Stage 00.25 配套练习：

```bash
cd /home/vipuser/Tilelang-Learning
python3 scripts/run_stage00_25_exercise.py --list
python3 scripts/run_stage00_25_exercise.py --task all
pytest tests/test_stage00_25_python_for_cpp_exercise.py -q
```

动手 Lab 入口：

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_starter_three_stage_lab
python3 scripts/run_starter_lab.py --lab all

cd /home/vipuser/Tilelang-Learning/projects/tilelang_three_stage_lab
python3 scripts/run_lab.py --lab all
python3 scripts/demo_common_errors.py --case all
```

## 毕业接口

`kernels.flash_attention.flash_attention_forward(q, k, v, causal=False, sm_scale=None)`

- 输入约定：CUDA tensor，形状 `(B, H, S, D)`。
- dtype 目标：优先支持 `float16`、`bfloat16`，内部按 `float32` 做数值稳定 reference。
- 维度目标：`D=64/128` 是主要优化目标，`S` 需要支持非 block 整除。
- 当前状态：接口已经固定，默认实现为 PyTorch online softmax reference；Stage 07 替换为 TileLang kernel。
- GEMM 入门 kernel 使用 `128x128x32` tile 作为 A100 上的可靠 baseline。

## 日常工作流

1. 先读当前 stage 文档中的阅读材料和概念清单。
2. 在 `notes/` 或 `reports/` 写下阶段问题、结论和实验记录。
3. 在 `kernels/` 写最小实现，先 PyTorch reference，再 TileLang kernel。
4. 在 `tests/` 加 correctness 对照，优先覆盖小 shape、边界 shape、非整除 shape。
5. 在 `benchmarks/` 记录 latency、吞吐、配置和硬件信息。
6. 阶段结束时用 `notes/operator_checklist.md` 自检。

## 验证命令

默认 correctness：

```bash
pytest
```

Stage 00 smoke：

```bash
python3 scripts/stage00_smoke.py
python3 scripts/check_learning_project.py
```

TileLang 编译 smoke：

```bash
RUN_TILELANG_SMOKE=1 pytest -m tilelang
```

Benchmark smoke：

```bash
python3 -m benchmarks.bench_gemm --m 128 --n 128 --k 128 --warmup 1 --repeat 2
python3 -m benchmarks.bench_flash_attention --batch 1 --heads 1 --seq 32 --dim 64 --warmup 1 --repeat 2
```

## 参考资料

- TileLang Installation: https://tilelang.com/get_started/Installation.html
- TileLang Language Basics: https://tilelang.com/programming_guides/language_basics.html
- TileLang Autotuning: https://tilelang.com/programming_guides/autotuning.html
- TileLang Debugging: https://tilelang.com/tutorials/debug_tools_for_tilelang.html
- TileLang Examples: https://github.com/tile-ai/tilelang/tree/main/examples
- PyPI: https://pypi.org/project/tilelang/
