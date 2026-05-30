# Tensor Shapes

shape 是调试 TileLang 和 attention 的第一线索。

更完整的逐步图解见 `docs/shape_walkthroughs.md`。

## 基础层

```text
vector_add:
a.shape = (N,)
b.shape = (N,)
c.shape = (N,)

row_sum:
x.shape = (M, N)
y.shape = (M,)
```

## 进阶层

```text
gemm:
A.shape = (M, K)
B.shape = (K, N)
C.shape = (M, N)

row_softmax:
x.shape = (M, N)
y.shape = (M, N)

rmsnorm:
x.shape = (M, N)
weight.shape = (N,)
y.shape = (M, N)
```

## Decoder Block

默认：

```text
x.shape = (B, S, hidden)
hidden = num_heads * head_dim
```

QKV：

```text
x2d.shape = (B*S, hidden)
qkv.shape = (B*S, 3*hidden)
q.shape = (B, num_heads, S, head_dim)
k.shape = (B, num_heads, S, head_dim)
v.shape = (B, num_heads, S, head_dim)
```

Attention：

```text
k.transpose(-2, -1).shape = (B, num_heads, head_dim, S)
scores.shape = (B, num_heads, S, S)
probs.shape = (B, num_heads, S, S)
out.shape = (B, num_heads, S, head_dim)
out_merged.shape = (B, S, hidden)
```

LM Head：

```text
block_out.shape = (B, S, hidden)
logits.shape = (B, S, vocab_size)
```

## 调试习惯

遇到错误先打印：

```python
print(x.shape, x.dtype, x.device, x.stride(), x.is_contiguous())
```

TileLang kernel 通常假设 contiguous。`transpose` 后经常需要 `.contiguous()`。

动手练习：

```bash
python3 scripts/run_lab.py --lab decoder
python3 scripts/demo_common_errors.py --case decoder_shape
```
