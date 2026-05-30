# row_sum

## 功能

`row_sum` 对二维 tensor 的每一行求和，是 reduction 的第一课。

## 公式

```text
y[m] = sum_n x[m, n]
```

## 输入输出 Shape

```text
x.shape = (M, N)
y.shape = (M,)
```

## PyTorch Reference

代码入口：`tilelab/basic.py::row_sum_reference`。

```python
return x.float().sum(dim=-1)
```

先转 fp32 是为了让参考结果更稳定。

## TileLang 数据流

```text
row m: X[m, 0:N] -> accumulator -> Y[m]
```

串行教学版中一个 block 负责一行，使用一个 thread 串行累加整行。

parallel 教学版中仍然一个 block 负责一行，但使用 `block_n` 个 thread 共同读取这一行，然后用 `T.reduce_sum` 汇总。

## 关键代码语义

- `T.Kernel(M, threads=1)`：每行一个 program。
- `T.alloc_fragment((1,), "float32")`：给当前行准备 fp32 累加器。
- `for col in T.serial(N)`：顺序访问这一行的所有列。
- `row_sum_parallel_tilelang` 使用 `values[col]` 保存每个 thread 读到的值，再调用 `T.reduce_sum(values, total)`。

## 边界条件

输入必须 rank-2。输出 dtype 固定为 fp32。

parallel v1 额外要求 `N <= block_n <= 1024`，且 `block_n` 是 2 的幂。无效列填 `0.0`，不会影响求和。

## 常见错误

- 把 reduction 当成 elementwise，多个 thread 同时写同一个输出。
- 忘记累加器清零。
- 直接用 fp16 累加导致误差偏大。

## 练习题

1. 手算一个 `x.shape=(2,3)` 的 row sum。
2. 解释为什么多个输入元素会合成一个输出元素。
3. 运行 `benchmarks.bench_reductions`，比较 `row_sum` 串行版和 parallel 版。
