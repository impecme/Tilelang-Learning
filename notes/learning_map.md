# TileLang-Learning Learning Map

这份地图把根目录笔记、练习脚本、starter 项目和 full lab 串起来。学习时不要急着跳到 Decoder，先把每一层的输入输出和验收问题跑通。

## 总路线

```text
Stage 00 环境与工程结构
  -> Stage 00.25 Python/PyTorch 最小基础
  -> Stage 00.5 Transformer/Attention 背景
  -> Starter Three Stage Lab
  -> Full Three Stage Lab
  -> Benchmark / Reduction / Decoder 深入优化
```

## 第一段：根目录基础

| 阶段 | 目标 | 必跑命令 | 产出 |
| --- | --- | --- | --- |
| Stage 00 | 确认环境和测试闭环 | `python3 scripts/check_env.py`、`pytest -q` | 知道 Python/PyTorch/TileLang 版本 |
| Stage 00.25 | 补齐 Python/PyTorch/pytest 基础 | `python3 scripts/run_stage00_25_exercise.py --task all` | 能解释 tensor 元信息和 reference test |
| Stage 00.5 | 理解 Transformer shape | 阅读 `notes/stage00_5_transformer_basics.md` | 能手推 Q/K/V 和 attention score shape |

## 第二段：轻量三阶段 starter

入口：`projects/tilelang_starter_three_stage_lab`

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_starter_three_stage_lab
python3 scripts/run_starter_lab.py --lab all
pytest -q
```

学习重点：

- 第一颗 kernel：`vector_add`
- 第一个矩阵 kernel：固定 tile-aligned GEMM
- 第一个模型流：`x -> linear -> GELU -> linear -> residual -> lm_head`

starter 的定位是“少而清楚”，适合第一次建立直觉。

## 第三段：完整版 three stage lab

入口：`projects/tilelang_three_stage_lab`

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_three_stage_lab
python3 scripts/run_lab.py --lab all
python3 scripts/demo_common_errors.py --case all
pytest -q
```

学习重点：

- Stage 01：基础 kernel、边界保护、dtype/device/contiguous 检查。
- Stage 02：GEMM、softmax、RMSNorm、reduction、benchmark。
- Stage 03：单 Decoder Block + logits 的 shape 流水线。

## 第四段：性能学习

先生成 CSV，再解读 CSV：

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_three_stage_lab
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```

每次性能实验至少写下：

- shape 和 dtype 是什么。
- PyTorch、TileLang serial、TileLang parallel 谁更快。
- 如果 parallel 没有更快，可能原因是什么。
- 下一轮只改变一个什么变量。

## 自检命令

根目录：

```bash
cd /home/vipuser/Tilelang-Learning
python3 scripts/check_learning_project.py
python3 scripts/run_stage00_25_exercise.py --task all
pytest -q
```

进入项目后：

```bash
python3 scripts/check_starter.py
python3 scripts/check_project.py
```

