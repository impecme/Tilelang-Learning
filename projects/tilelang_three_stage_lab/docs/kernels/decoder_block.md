# decoder_block

## 功能

`decoder_block` 把前面学过的算子串成一个迷你 Transformer Decoder Block。

## 公式/流程

```text
x
 -> RMSNorm
 -> QKV Linear
 -> Causal Attention
 -> Output Linear
 -> Residual
 -> RMSNorm
 -> MLP Up + GELU
 -> MLP Down
 -> Residual
```

`mini_inference` 还会追加 LM Head：

```text
block_out @ lm_head_weight + lm_head_bias -> logits
```

## 输入输出 Shape

默认 tiny 教学 shape：

```text
x.shape = (1, 128, 128)
num_heads = 2
head_dim = 64
logits.shape = (1, 128, 128)
```

默认完整教学 shape：

```text
x.shape = (1, 128, 256)
num_heads = 4
head_dim = 64
logits.shape = (1, 128, 4096)
```

## PyTorch Reference

代码入口：

- `tilelab/decoder.py::decoder_block_reference`
- `tilelab/decoder.py::mini_inference_reference`

Reference 用 PyTorch matmul、softmax、GELU 串出标准答案。

## TileLang 数据流

TileLang 版本用多个教学 kernel 编排：

- linear：`gemm_tilelang`
- norm：`rmsnorm_tilelang`
- optimized norm：`rmsnorm_parallel_tilelang`
- attention mask：`causal_mask_tilelang`
- softmax：`row_softmax_tilelang`
- optimized softmax：`row_softmax_parallel_tilelang`
- residual：`vector_add_tilelang`
- MLP activation：`linear_bias_gelu_tilelang`

Python 只负责 reshape、split QKV、循环 batch/head。

`decoder_block_tilelang_optimized` 和 `mini_inference_tilelang_optimized` 复用同一条流水线，只把 RMSNorm 和 attention softmax 替换成 parallel reduction 教学版。

## 关键代码语义

- `MiniDecoderConfig` 固定 hidden/head/seq/vocab 等学习 shape。
- `MiniDecoderWeights` 保存所有权重。
- `_split_qkv` 把 `(B,S,3*hidden)` 拆成 `(B,H,S,D)`。
- `_attention_tilelang` 对每个 head 做 `QK^T -> mask -> softmax -> P@V`。
- optimized 入口不改变数学公式，只改变 reduction kernel 的实现方式。

## 边界条件

- `hidden_size == num_heads * head_dim`。
- 权重 shape、dtype、device 必须和 config、输入一致。
- GEMM v1 需要 tile-aligned shape。
- parallel reduction v1 要求被 reduction 的列数 `<= block_n <= 1024`。
- 不做 tokenizer、采样、训练、KV cache。

## 常见错误

- 把 QKV shape 顺序搞混。
- 忘记 causal mask 是屏蔽未来位置。
- residual add 没有和原输入对齐。
- 权重放在 CPU，输入放在 CUDA。

## 练习题

1. 写出 `QK^T` 的 shape 推导。
2. 解释为什么 attention 输出要从 `(B,H,S,D)` merge 回 `(B,S,hidden)`。
3. 对比 `decoder_block_tilelang` 和 `decoder_block_tilelang_optimized` 的 benchmark CSV。
4. 思考 KV cache 会改变哪一部分数据流。
