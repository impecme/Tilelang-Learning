# Lab：Decoder Shape Trace

## 目标

从 `x=(B,S,H)` 一路追踪到 `logits=(B,S,V)`，理解 Decoder Block 的张量流。

## 要读的代码

1. `tilelab/decoder.py::MiniDecoderConfig`
2. `tilelab/decoder.py::decoder_block_reference`
3. `tilelab/decoder.py::decoder_block_tilelang`
4. `tilelab/decoder.py::decoder_block_tilelang_optimized`
5. `docs/shape_walkthroughs.md`

## 运行命令

```bash
python3 scripts/run_lab.py --lab decoder
python3 scripts/demo_common_errors.py --case decoder_shape
python3 scripts/demo_common_errors.py --case decoder_dtype
```

## 动手改法

- 写出 tiny config 下每一步 shape。
- 把 `hidden_size` 改成不等于 `num_heads * head_dim`，观察配置报错。
- 对比普通 TileLang 路径和 optimized reductions 路径。

## 自测问题

1. Q/K/V 的 shape 为什么是 `(B, heads, S, D)`？
2. `QK^T` 为什么得到 `(S,S)`？
3. logits 为什么最后一维是 `vocab_size`？

## 常见错误

- 混淆 `(B,S,H,D)` 和 `(B,H,S,D)`。
- 忘记 K 要转置。
- 以为 logits 已经是概率。
