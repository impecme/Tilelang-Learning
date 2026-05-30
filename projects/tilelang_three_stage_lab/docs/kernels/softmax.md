# row_softmax

## 功能

`row_softmax` 对二维 tensor 的每一行做 softmax，把分数变成权重。

## 公式

```text
m = max_j x[i, j]
y[i, j] = exp(x[i, j] - m) / sum_k exp(x[i, k] - m)
```

## 输入输出 Shape

```text
x.shape = (M, N)
y.shape = (M, N)
```

## PyTorch Reference

代码入口：`tilelab/advanced.py::row_softmax_reference`。

```python
return torch.softmax(x.float(), dim=-1).to(dtype=x.dtype)
```

## TileLang 数据流

```text
row -> max -> exp/sum -> normalize -> output row
```

串行教学版中每行一个 program，串行完成 max、sum、normalize 三步。

parallel 教学版中每行仍然一个 program，但 `block_n` 个 thread 共同处理一行。max 阶段用 `T.reduce_max`，sum 阶段用 `T.reduce_sum`。

## 关键代码语义

- `max_val[0] = -3.402823e38`：用极小值初始化 max。
- 第一轮 `T.serial(N)` 求行最大值。
- 第二轮求 exp 后的分母。
- 第三轮写出归一化结果。
- `row_softmax_parallel_tilelang` 先把无效列填成极小值参与 max，再把无效列填成 `0.0` 参与 sum。

## 边界条件

输入必须 rank-2。v1 没有处理全 mask 为 `-inf` 的特殊行，Decoder 中的 causal mask 保证每行至少能看自己。

parallel v1 额外要求 `N <= block_n <= 1024`，且 `block_n` 是 2 的幂。

## 常见错误

- 不减 max，导致 `exp` 溢出。
- 对错维度做 softmax。
- fp16 直接计算 reference，误差变大。

## 练习题

1. 手算 `[1, 2, 3]` 的 stable softmax。
2. 解释为什么减去 max 不改变 softmax 结果。
3. 思考 causal attention 中 softmax 的最后一维代表什么。
4. 跑 `benchmarks.bench_reductions`，比较 serial softmax 和 parallel softmax。
