# TileLang Starter Three Stage Lab

这是 `tilelang_three_stage_lab` 之前的轻量入门版。它只保留三件事：

- 能跑通第一个 TileLang GPU kernel。
- 能跑通第一个 TileLang GEMM。
- 能把几个小算子串成一个迷你模型流。

它故意没有完整版那么完备：没有逐 kernel 长文档、没有 benchmark 模板、没有阶段元数据、没有报告系统。你先用它建立第一轮直觉，再去学完整版。

## 快速开始

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_starter_three_stage_lab
python3 -m pip install -r requirements.txt
python3 scripts/check_starter.py
pytest -q
python3 scripts/run_starter_lab.py --lab all
```

可选 TileLang 编译 smoke：

```bash
RUN_TILELANG_SMOKE=1 pytest -q -m tilelang
```

## 三个轻量阶段

| 阶段 | 你只学什么 | 算子/流水线 | 进入下一阶段前要会什么 |
| --- | --- | --- | --- |
| `01_first_kernel` | 第一个 GPU kernel | `vector_add` | 能解释 `idx` 和 boundary guard |
| `02_first_matrix_op` | 第一个矩阵算子 | `gemm` | 能解释 `M/N/K` 和 tile-aligned shape |
| `03_first_model_flow` | 第一个模型数据流 | `linear -> GELU -> linear -> residual -> lm_head` | 能从 `x` 推导到 `logits` 的 shape |

阶段检查：

```bash
python3 scripts/check_starter.py --stage 1
python3 scripts/check_starter.py --stage 2
python3 scripts/check_starter.py --stage 3
```

## 动手 Lab

如果你不确定先读哪里，直接从 labs 开始：

```bash
python3 scripts/run_starter_lab.py --list
python3 scripts/run_starter_lab.py --lab all
python3 scripts/run_starter_lab.py --lab 1 --run-tilelang
```

Lab 文档入口：`labs/README.md`。

每个 lab 都包含：目标、要读的代码、运行命令、动手改法、必须回答的 3 个问题和常见错误。跑完后用 `labs/answer_key.md` 对照预期输出和参考答案。

## 目录结构

```text
tilelang_starter_three_stage_lab/
  01_first_kernel/        # 第一个 TileLang kernel
  02_first_matrix_op/     # 第一个 GEMM
  03_first_model_flow/    # 第一个迷你模型流
  labs/                   # 可运行动手练习
  starter_tilelab/        # 最小 Python package
  tests/                  # 最小 correctness tests
  scripts/                # 简单健康检查
```

## Starter 与 Full Lab 的关系

| 项目 | 定位 | 文档量 | 工程检查 | 适合什么时候学 |
| --- | --- | --- | --- | --- |
| `tilelang_starter_three_stage_lab` | 第一轮入门 | 少 | 简单 | 还没形成 TileLang 直觉时 |
| `tilelang_three_stage_lab` | 第二轮系统学习 | 多 | 严格 | 想系统掌握 kernel、shape、测试、benchmark 时 |

推荐顺序：

1. 先跑完 starter 三阶段。
2. 跑完 `python3 scripts/run_starter_lab.py --lab all`。
3. 再进入完整版 `../tilelang_three_stage_lab/docs/lab_index.md`。
4. 遇到 starter 里看不懂的概念，再回完整版的详细文档查。

## Public API

```python
from starter_tilelab.basic import vector_add_reference, vector_add_tilelang
from starter_tilelab.advanced import gemm_reference, gemm_tilelang
from starter_tilelab.model import (
    TinyModelConfig,
    TinyModelWeights,
    make_tiny_weights,
    tiny_model_reference,
    tiny_model_tilelang,
)
```

## 当前限制

- `gemm_tilelang` 默认使用 `16x16x32` 教学 tile，只支持 tile-aligned shape。
- `K == 32` 的 starter GEMM 路径使用 `T.gemm`；更大的 K 维先走串行教学 fallback，完整 pipelined GEMM 放到 full lab 学。
- tiny model 没有 attention、norm、bias、tokenizer、采样和训练。
- TileLang 路径要求 CUDA tensor 且 contiguous。
- correctness 优先，性能不作为本项目目标。
