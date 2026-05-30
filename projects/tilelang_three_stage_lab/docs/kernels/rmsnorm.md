# rmsnorm

## 功能

`rmsnorm` 是 LLM decoder block 常见归一化算子。

## 公式

```text
variance = mean(x^2)
y = x * rsqrt(variance + eps) * weight
```

## 输入输出 Shape

```text
x.shape = (M, N)
weight.shape = (N,)
y.shape = (M, N)
```

## PyTorch Reference

代码入口：`tilelab/advanced.py::rmsnorm_reference`。

```python
variance = x.float().pow(2).mean(dim=-1, keepdim=True)
out = x.float() * torch.rsqrt(variance + eps) * weight.float()
```

## TileLang 数据流

```text
row -> sum of squares -> rsqrt scale -> multiply weight -> output row
```

串行教学版中每行一个 program，串行统计平方和。

parallel 教学版中每行一个 program，`block_n` 个 thread 分别计算 `x^2`，再用 `T.reduce_sum` 得到平方和。

## 关键代码语义

- `ss` 是 fp32 平方和。
- `scale = T.rsqrt(ss / N + eps)`。
- 输出逐列乘以 `weight[col]`。
- `rmsnorm_parallel_tilelang` 的平方和也使用 fp32 fragment，输出仍回到输入 dtype。

## 边界条件

`weight.shape` 必须等于 `(N,)`。输入必须 rank-2。

parallel v1 额外要求 `N <= block_n <= 1024`，且 `block_n` 是 2 的幂。

## 常见错误

- 和 LayerNorm 混淆。RMSNorm 不减均值。
- 忘记 `eps`，小数值时可能不稳定。
- weight shape 写成 `(M,)`。

## 练习题

1. 对 `[3, 4]` 手算 RMS。
2. 解释 RMSNorm 为什么只需要一组 `weight`。
3. 思考为什么 LLM 中常在 attention 前后使用 norm。
4. 跑 `benchmarks.bench_reductions`，比较 serial RMSNorm 和 parallel RMSNorm。
