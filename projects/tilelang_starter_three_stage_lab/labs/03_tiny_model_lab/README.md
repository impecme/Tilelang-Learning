# Lab 3：Tiny Model Flow

## 目标

跟踪极简模型流：

```text
x -> linear -> GELU -> linear -> residual -> lm_head -> logits
```

它不是完整 Decoder Block，没有 attention。attention 放到 full lab 学。

## 要读的代码

1. `starter_tilelab/model.py::TinyModelConfig`
2. `starter_tilelab/model.py::make_tiny_weights`
3. `starter_tilelab/model.py::tiny_model_reference`
4. `starter_tilelab/model.py::tiny_model_tilelang`

## 运行命令

```bash
python3 scripts/run_starter_lab.py --lab 3
python3 scripts/run_starter_lab.py --lab 3 --run-tilelang
```

## 动手改法

- 写出默认 `x.shape` 和 `logits.shape`。
- 把 `vocab_size` 改小，观察 logits 最后一维如何变化。
- 故意传错 `x.shape`，观察报错中期望 shape 是什么。

## 必须回答的 3 个问题

1. residual add 为什么要求 `x` 和 projected shape 一样？
2. logits 的最后一维为什么等于 `vocab_size`？
3. 这个 starter 模型和 full Decoder Block 最大区别是什么？

## 常见错误

- 把 `seq_len` 和 `hidden_size` 维度写反。
- 忘记权重也要和输入在同一 device。
- 误以为 logits 已经是概率。

## 下一阶段入口

完成后进入完整版 `tilelang_three_stage_lab`，从 `docs/lab_index.md` 开始。
