# Starter Labs

starter labs 是第一轮入门练习。目标是少而清楚：每个 lab 只让你读一个核心文件、跑一个命令、改一个小点、回答三个问题。

## 推荐顺序

| Lab | 主题 | 代码入口 | 目标 |
| --- | --- | --- | --- |
| 1 | 第一个 kernel | `starter_tilelab/basic.py` | 读懂 `idx` 和 boundary guard |
| 2 | 第一个矩阵算子 | `starter_tilelab/advanced.py` | 读懂 `M/N/K` 和 tile-aligned shape |
| 3 | 第一个模型流 | `starter_tilelab/model.py` | 从 `x` 跟踪到 `logits` |

## 命令

```bash
python3 scripts/run_starter_lab.py --list
python3 scripts/run_starter_lab.py --lab all
python3 scripts/run_starter_lab.py --lab 1 --run-tilelang
```

默认命令只跑 PyTorch reference 和轻量 shape 检查。`--run-tilelang` 才会编译 TileLang kernel。

参考答案和预期输出：`labs/answer_key.md`。

## 学完后进入 Full Lab

完成三个 starter labs 后，去完整版：

```bash
cd /home/vipuser/Tilelang-Learning/projects/tilelang_three_stage_lab
python3 scripts/run_lab.py --list
```
