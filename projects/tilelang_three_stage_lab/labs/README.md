# Full Lab Hands-On Index

这些 labs 是完整版课程的动手层。它们把文档、代码、测试、benchmark 串起来，让你每次只聚焦一个学习动作。

## 推荐顺序

| Lab | 主题 | 对应阶段 | 目标 |
| --- | --- | --- | --- |
| basic | kernel debug | Stage 01 | 学会定位 shape/device/contiguous/boundary 问题 |
| gemm | tile shape | Stage 02 | 学会解释 GEMM tile 对齐限制 |
| reduction | parallel reduction | Stage 01/02 | 对比串行和并行 reduction |
| decoder | shape trace | Stage 03 | 从 `x` 跟踪到 `logits` |
| benchmark | reading CSV | Stage 02/03 | 读懂 latency、backend、notes |

## 命令

```bash
python3 scripts/run_lab.py --list
python3 scripts/run_lab.py --lab all
python3 scripts/run_lab.py --lab reduction --run-tilelang
python3 scripts/demo_common_errors.py --case all
```

默认只跑轻量 reference 和 shape 检查。`--run-tilelang` 才编译 TileLang kernel。

## 和文档的关系

- 学习路线：`docs/learning_path.md`
- Lab 说明：`docs/lab_index.md`
- 参考答案：`docs/lab_answer_key.md`
- Shape 图解：`docs/shape_walkthroughs.md`
- 错误图鉴：`docs/error_gallery.md`
