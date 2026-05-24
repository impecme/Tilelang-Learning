# Tilelang-Learning

这是一个从 TileLang 入门到 AI 算子开发的学习工程。目标不是只读文档，而是每周留下可运行代码、correctness test、benchmark 记录和复盘笔记，最终完成一个 FlashAttention forward 原型。

## 环境基线

- 目标机器：NVIDIA A100，CUDA/PyTorch 环境优先。
- TileLang 版本：`0.1.9`。
- PyTorch：使用当前系统环境中已经安装的 CUDA 版 PyTorch。
- 系统 `nvcc` 与 PyTorch CUDA 版本可能不同，因此第一阶段优先用 PyPI wheel，不做源码构建。

安装与检查：

```bash
cd /home/vipuser/Tilelang-Learning
python3 -m pip install -r requirements.txt
python3 scripts/check_env.py
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
  notes/          # 10 周学习笔记
  reports/        # benchmark 输出和复盘报告
  scripts/        # 环境检查和 smoke test
```

## 概念导读

开始写 kernel 前先读 [notes/concepts.md](notes/concepts.md)。它介绍 AI 算子、TileLang 的位置、GPU kernel/block/thread、shape/layout/stride、内存层级、tiling、GEMM、softmax、FlashAttention、correctness 和 benchmark 等核心概念。

推荐顺序：

1. 先读 `concepts.md` 的 1-6 节。
2. 跑通 Week 01 的环境和 vector add。
3. Week 03-04 回看 GEMM 和内存层级部分。
4. Week 07 前重点复习 online softmax 和 FlashAttention。

## 毕业接口

`kernels.flash_attention.flash_attention_forward(q, k, v, causal=False, sm_scale=None)`

- 输入：CUDA tensor，形状 `(B, H, S, D)`。
- dtype：优先支持 `float16`、`bfloat16`，内部按 `float32` 做数值稳定 reference。
- 维度：`D=64/128` 是主要优化目标，`S` 需要支持非 block 整除。
- 当前阶段：接口已经固定，默认实现为 PyTorch online softmax reference；第 8-9 周替换为 TileLang kernel。
- GEMM 入门 kernel 使用 `128x128x32` tile 作为 A100 上的可靠 baseline。

## 10 周路线

| Week | 主题 | 产物 |
| --- | --- | --- |
| 01 | 环境与最小闭环 | TileLang import、vector add、项目测试与 benchmark 框架 |
| 02 | TileLang DSL 核心 | `T.Kernel`、`T.Parallel`、`T.serial`、`T.unroll`、动态 shape 练习 |
| 03 | 内存层级与调试 | global/shared/fragment、`T.copy`、IR/debug 笔记、tiled matmul v0 |
| 04 | GEMM 性能主线 | `T.gemm`、`T.Pipelined`、tile size 对比 |
| 05 | Autotuning 与 benchmark | 统一 benchmark harness、保存最佳配置 |
| 06 | 常见 AI 算子模式 | matmul+bias+activation、softmax、layernorm/RMSNorm 中至少两个 |
| 07 | Attention 基线 | PyTorch reference、naive attention、online softmax 推导 |
| 08 | FlashAttention forward v1 | non-causal TileLang FlashAttention forward |
| 09 | 性能调优与 causal | block/threads/stages 调参，对比 PyTorch SDPA，可选 causal |
| 10 | 收尾复盘 | API 固化、benchmark 报告、算子开发 checklist |

## 日常工作流

1. 在 `notes/weekXX.md` 写当天目标、疑问、结论。
2. 在 `kernels/` 写最小实现，先 PyTorch reference，再 TileLang kernel。
3. 在 `tests/` 加 correctness 对照，优先覆盖边界 shape。
4. 在 `benchmarks/` 记录 latency、吞吐、配置和硬件信息。
5. 每周结束把结果写入 `reports/`。

## 参考资料

- TileLang Installation: https://tilelang.com/get_started/Installation.html
- TileLang Language Basics: https://tilelang.com/programming_guides/language_basics.html
- TileLang Autotuning: https://tilelang.com/programming_guides/autotuning.html
- TileLang Debugging: https://tilelang.com/tutorials/debug_tools_for_tilelang.html
- TileLang Examples: https://github.com/tile-ai/tilelang/tree/main/examples
- PyPI: https://pypi.org/project/tilelang/
