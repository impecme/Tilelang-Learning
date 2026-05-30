# 03 First Model Flow

## 目标

把前两阶段的小算子串成第一个迷你模型流：

```text
x -> linear -> GELU -> linear -> residual -> lm_head -> logits
```

这里没有 attention，也不是完整 Decoder Block。attention、RMSNorm、QKV 和 causal mask 都放到完整版项目学习。

## 要读的代码

- `starter_tilelab/model.py`
- `tests/test_model.py`

默认 shape：

```text
x = (seq_len, hidden_size) = (16, 64)
fc1_weight = (64, 128)
fc2_weight = (128, 64)
lm_head_weight = (64, 256)
logits = (16, 256)
```

## 运行命令

```bash
python3 scripts/check_starter.py --stage 3
pytest tests/test_model.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_model.py -q -m tilelang
```

## 必须回答的 3 个问题

1. `x @ fc1_weight` 的输出 shape 是什么？
2. residual add 为什么要求两个 tensor shape 相同？
3. `logits=(seq_len, vocab_size)` 里的 `vocab_size` 代表什么？

## 下一阶段入口

能从 `x` 推导到 `logits` 后，进入完整版 `../tilelang_three_stage_lab/README.md`。
