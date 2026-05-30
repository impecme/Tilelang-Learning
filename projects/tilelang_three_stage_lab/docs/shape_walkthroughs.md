# Shape Walkthroughs

这份文档只讲 shape。读 Decoder 或 GEMM 时，如果脑子里维度开始打结，先回到这里。

## Vector Add

```text
a.shape = (N,)
b.shape = (N,)
out.shape = (N,)
```

每个输出元素只依赖同一个位置的两个输入：

```text
out[i] = a[i] + b[i]
```

## Row Sum

```text
x.shape = (M, N)
y.shape = (M,)
```

每一行变成一个标量：

```text
y[m] = sum(x[m, 0:N])
```

## GEMM

```text
A.shape = (M, K)
B.shape = (K, N)
C.shape = (M, N)
```

K 是累加维，必须在 A 和 B 中相等。

## Attention

默认 tiny config：

```text
B = 1
S = 128
hidden_size = 128
num_heads = 2
head_dim = 64
```

流程：

```text
x:        (B, S, hidden)
qkv:      (B, S, 3 * hidden)
q/k/v:    (B, heads, S, head_dim)
scores:   (B, heads, S, S)
probs:    (B, heads, S, S)
attn_out: (B, S, hidden)
```

`scores` 的两个 `S` 分别表示 query 位置和 key 位置。

## MLP 和 LM Head

```text
norm2:      (B*S, hidden)
mlp_up:     (B*S, ffn_hidden)
mlp_down:   (B*S, hidden)
block_out:  (B, S, hidden)
logits:     (B, S, vocab_size)
```

logits 不是概率。它是每个位置对词表中每个 token 的原始分数。

## 自测

1. 如果 `hidden_size=256, num_heads=4`，`head_dim` 应该是多少？
2. 如果 `vocab_size=4096`，logits 最后一维是多少？
3. 为什么 residual add 前后 shape 必须一样？
