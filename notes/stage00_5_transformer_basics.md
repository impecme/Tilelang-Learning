# Stage 00.5 - 深度学习与 Transformer 最小背景

## 阶段目标

这一阶段，我专门补齐进入 Attention/FlashAttention 前必须懂的深度学习背景。目标不是系统学完深度学习，而是看懂这句话：

```text
attention 的 QK^T -> softmax -> PV
```

学完这个阶段后，应能解释：

- `Q`、`K`、`V` 分别是什么。
- 为什么有 `K^T`。
- `QK^T` 的 shape 为什么是 `(S, S)`。
- softmax 在 attention 里做什么。
- `PV` 更准确地说应该写成 `P @ V`，其中 `P = softmax(QK^T / sqrt(D))`。

## 先修状态

- 会 Python 和一点 PyTorch。
- 知道矩阵乘法大概是什么。
- 如果 Python/PyTorch 不熟，先完成 `notes/stage00_25_python_for_cpp.md`。
- 暂时不要求已经学过 Transformer。

## 阅读

- 本文件完整阅读。
- `notes/concepts_deep_dive.md` 第 1、2、7、11 节。
- `kernels/reference.py` 中 `naive_attention_forward`。
- PyTorch `torch.matmul` 和 `torch.softmax` 的基本用法。

## 1. 深度学习最小图景

深度学习可以先粗略理解为：用很多层可训练函数，把输入数据变成输出结果。

例如文本模型中：

```text
文字/token -> 向量 embedding -> 多层神经网络 -> 输出
```

一个 token 不能直接被神经网络计算，所以要先变成向量。比如一句话有 `S` 个 token，每个 token 用一个 `D` 维向量表示，那么输入可以写成：

```text
X.shape = (S, D)
```

如果有 batch：

```text
X.shape = (B, S, D)
```

含义：

- `B`：batch size，一次处理多少条样本。
- `S`：sequence length，一条样本有多少个 token。
- `D`：hidden dimension，每个 token 用多少维向量表示。

## 2. 矩阵乘法必须懂到什么程度

矩阵乘法的 shape 规则：

```text
A.shape = (M, K)
B.shape = (K, N)
C = A @ B
C.shape = (M, N)
```

中间维度 `K` 必须相同。每个输出元素：

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

例子：

```text
A: (2, 3)
B: (3, 4)
A @ B: (2, 4)
```

如果 `K.shape = (S, D)`，那么 `K^T.shape = (D, S)`。因此：

```text
Q.shape = (S, D)
K.shape = (S, D)
Q @ K^T = (S, D) @ (D, S) = (S, S)
```

这就是 attention 里 `QK^T` 会得到 token-to-token 分数矩阵的原因。

## 3. Token 为什么需要互相看

Transformer 的核心是：一个 token 的表示不应该只由它自己决定，还应该参考上下文。

比如句子：

```text
我 喜欢 苹果
```

“苹果”可能是水果，也可能是公司。模型需要根据上下文判断。Self-attention 做的事情就是让每个 token 都去看其它 token，然后决定应该从哪些 token 里取信息。

可以先把 attention 理解成一个“按相关性加权汇总信息”的过程：

```text
当前 token 想找什么 -> 和所有 token 匹配 -> 得到权重 -> 按权重汇总信息
```

这句话对应：

```text
Query -> Key -> softmax weights -> Value weighted sum
```

## 4. Q、K、V 是什么

给定输入 token 表示：

```text
X.shape = (S, D_model)
```

Transformer 会用三个线性变换得到：

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

它们的含义可以这样记：

- `Q` / Query：当前 token 想找什么信息。
- `K` / Key：每个 token 提供一个“可匹配标签”。
- `V` / Value：每个 token 真正携带、要被汇总的信息。

直觉类比：

```text
Query: 想问的问题
Key: 每条资料的标签
Value: 每条资料的正文内容
```

Attention 做的是：

```text
用 Query 去匹配所有 Key，得到每条 Value 应该占多少权重。
```

## 5. QK^T 在算什么

假设：

```text
Q.shape = (S, D)
K.shape = (S, D)
```

那么：

```text
scores = Q @ K^T
scores.shape = (S, S)
```

`scores[i, j]` 表示：

```text
第 i 个 token 的 Query
和第 j 个 token 的 Key
有多匹配
```

也就是：

```text
scores[i, j] = dot(Q[i], K[j])
```

如果 `scores[i, j]` 大，说明第 `i` 个 token 更应该关注第 `j` 个 token。

为什么要转置 `K`？

因为 `Q` 和 `K` 原本都是 `(S, D)`。想让每个 `Q[i]` 和每个 `K[j]` 做点积，就需要：

```text
(S, D) @ (D, S) -> (S, S)
```

所以要用 `K^T`。

## 6. 为什么要除以 sqrt(D)

标准 attention 通常写成：

```text
scores = Q @ K^T / sqrt(D)
```

原因：当 `D` 比较大时，点积值可能变得很大。点积值太大后，softmax 会变得非常尖锐，也就是一个位置接近 1，其它位置接近 0，训练会不稳定。

除以 `sqrt(D)` 是一种缩放，让 scores 的数值范围更稳定。

在代码里通常叫：

```text
scale = 1 / sqrt(D)
scores = Q @ K^T * scale
```

## 7. softmax 在 attention 里做什么

`QK^T` 得到的是任意实数分数，比如：

```text
scores = [2.0, 1.0, -1.0]
```

它们还不是权重。权重应该满足：

```text
每个值 >= 0
所有值加起来 = 1
```

softmax 会把分数变成概率式权重：

```text
weights = softmax(scores)
```

如果某一行 scores 是：

```text
[2.0, 1.0, -1.0]
```

softmax 后可能类似：

```text
[0.70, 0.26, 0.04]
```

在 attention 里：

```text
P = softmax(QK^T / sqrt(D))
P.shape = (S, S)
```

`P[i, j]` 表示第 `i` 个 token 从第 `j` 个 token 取信息的权重。

## 8. P @ V 在算什么

设：

```text
P.shape = (S, S)
V.shape = (S, D)
out = P @ V
out.shape = (S, D)
```

第 `i` 个输出 token：

```text
out[i] = sum_j P[i, j] * V[j]
```

意思是：

```text
第 i 个 token 根据 attention 权重，从所有 Value 中加权汇总信息。
```

所以完整流程是：

```text
scores = Q @ K^T / sqrt(D)
P = softmax(scores)
out = P @ V
```

有时口头说成：

```text
QK^T -> softmax -> PV
```

更严格写法应该是：

```text
QK^T -> softmax -> P @ V
```

其中 `P` 是 softmax 后的 attention probability。

## 9. 一个极小数字例子

假设有 2 个 token，每个向量 2 维：

```text
Q = [[1, 0],
     [0, 1]]

K = [[1, 0],
     [1, 1]]

V = [[10, 0],
     [0, 20]]
```

先算：

```text
K^T = [[1, 1],
       [0, 1]]
```

所以：

```text
Q @ K^T =
[[1, 1],
 [0, 1]]
```

解释：

- 第 0 个 token 的 query 和两个 key 都匹配，分数都是 1。
- 第 1 个 token 的 query 更匹配第 1 个 key。

对每一行做 softmax：

```text
P = softmax(scores, dim=-1)
```

大致会得到：

```text
P[0] = [0.5, 0.5]
P[1] = [0.27, 0.73]
```

最后：

```text
out = P @ V
```

第 0 个输出：

```text
out[0] = 0.5 * [10, 0] + 0.5 * [0, 20] = [5, 10]
```

第 1 个输出：

```text
out[1] = 0.27 * [10, 0] + 0.73 * [0, 20] = [2.7, 14.6]
```

这就是 attention：按相关性从所有 token 的 value 中取信息。

## 10. PyTorch 最小代码

```python
import math
import torch

torch.manual_seed(0)

S = 4
D = 8

Q = torch.randn(S, D)
K = torch.randn(S, D)
V = torch.randn(S, D)

scores = Q @ K.T / math.sqrt(D)
P = torch.softmax(scores, dim=-1)
out = P @ V

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)
print("scores:", scores.shape)
print("P:", P.shape)
print("out:", out.shape)
print("row sums:", P.sum(dim=-1))
```

输出应类似：

```text
Q: torch.Size([4, 8])
K: torch.Size([4, 8])
V: torch.Size([4, 8])
scores: torch.Size([4, 4])
P: torch.Size([4, 4])
out: torch.Size([4, 8])
row sums: tensor([1., 1., 1., 1.])
```

`P` 的每一行加起来是 1，因为每一行表示“当前 token 如何分配注意力到所有 token”。

## 11. 加上 batch 和 head

真实模型通常不是 `(S, D)`，而是：

```text
Q.shape = (B, H, S, D)
K.shape = (B, H, S, D)
V.shape = (B, H, S, D)
```

含义：

- `B`：batch。
- `H`：attention heads。
- `S`：sequence length。
- `D`：每个 head 的维度。

这时 attention 是对每个 `(b, h)` 独立做：

```text
scores[b, h] = Q[b, h] @ K[b, h].T
P[b, h] = softmax(scores[b, h])
out[b, h] = P[b, h] @ V[b, h]
```

shape：

```text
Q[b,h]: (S, D)
K[b,h].T: (D, S)
scores[b,h]: (S, S)
P[b,h]: (S, S)
V[b,h]: (S, D)
out[b,h]: (S, D)
```

整体：

```text
scores.shape = (B, H, S, S)
out.shape = (B, H, S, D)
```

## 12. Self-Attention 和 Cross-Attention

Self-attention：

```text
Q, K, V 都来自同一个 X
```

例如：

```text
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

每个 token 看同一句话里的其它 token。

Cross-attention：

```text
Q 来自一组 token
K, V 来自另一组 token
```

例如 decoder 里的 token 去看 encoder 的输出。

本工程的 FlashAttention forward 优先关心 self-attention，并且第一版假设 `q/k/v` 的 sequence length 相同。

## 13. Multi-Head Attention 是什么

单个 attention head 可以看成一种“看上下文的方式”。Multi-head attention 是同时用多个 head：

```text
head 0 看语法关系
head 1 看位置关系
head 2 看实体关系
...
```

工程上常见 shape：

```text
X: (B, S, D_model)
Q/K/V after projection: (B, S, H * D)
reshape: (B, S, H, D)
transpose: (B, H, S, D)
```

TileLang kernel 通常希望输入已经是：

```text
(B, H, S, D)
```

这样每个 head 可以独立处理。

## 14. Mask 是什么

Mask 用来禁止某些位置被 attention 到。

最常见的是 causal mask，用于自回归语言模型：

```text
第 i 个 token 只能看 0..i 的 token，不能看未来 token
```

如果 `j > i`，就把 score 设成 `-inf`：

```text
scores[i, j] = -inf
```

softmax 后该位置权重变成 0。

例子，`S=4`：

```text
允许位置:
row 0: 0
row 1: 0,1
row 2: 0,1,2
row 3: 0,1,2,3
```

对应矩阵：

```text
[[ok, no, no, no],
 [ok, ok, no, no],
 [ok, ok, ok, no],
 [ok, ok, ok, ok]]
```

本工程第一版 FlashAttention 先做 non-causal，因为 causal mask 会增加 block 内边界判断复杂度。

## 15. 为什么 FlashAttention 有价值

普通 attention 会显式生成：

```text
scores.shape = (B, H, S, S)
P.shape = (B, H, S, S)
```

当 `S` 很大时，这两个矩阵非常大。

例如：

```text
B = 1
H = 16
S = 4096
```

则：

```text
scores elements = 1 * 16 * 4096 * 4096 = 268,435,456
```

如果是 fp16：

```text
268,435,456 * 2 bytes ≈ 512 MB
```

只保存一个 scores 就约 512 MB。再加 probs、读写带宽和训练中的反向缓存，成本会非常高。

FlashAttention 的思想：

```text
不完整保存 scores 和 P
分块处理 K/V
边算 softmax 边更新输出
只把最终 out 写回显存
```

所以它主要优化：

- 显存占用。
- global memory 读写。
- 长序列 attention 的性能。

## 16. 现在需要懂到什么程度

进入 TileLang attention kernel 前，不需要完整学完 Transformer，但必须能回答：

- `Q.shape = (B,H,S,D)` 是什么意思？
- `K^T` 为什么把 `(S,D)` 变成 `(D,S)`？
- `Q @ K^T` 为什么得到 `(S,S)`？
- `scores[i,j]` 表示什么？
- softmax 为什么按最后一维做？
- `P[i,j]` 表示什么？
- `out = P @ V` 为什么回到 `(S,D)`？
- causal mask 为什么把未来位置设成 `-inf`？
- FlashAttention 为什么不想保存完整 `(S,S)`？

如果这些都能答出来，再进入 Stage 06 会舒服很多。

## 17. 小练习

1. Shape 练习：
   - `Q.shape = (128, 64)`
   - `K.shape = (128, 64)`
   - `V.shape = (128, 64)`
   - 写出 `Q @ K.T`、`softmax` 后的 `P`、`P @ V` 的 shape。

2. Batch/head 练习：
   - `Q.shape = (2, 8, 512, 64)`
   - 写出 `scores` 和 `out` 的 shape。

3. 代码练习：
   - 用 PyTorch 手写一次 attention。
   - 打印每个中间 tensor 的 shape。
   - 检查 `P.sum(dim=-1)` 是否接近 1。

4. 解释练习：
   - 用自己的话解释 `QK^T -> softmax -> P@V`。
   - 不允许只背公式，要说出每一步在“找信息、分配权重、汇总信息”中的角色。
