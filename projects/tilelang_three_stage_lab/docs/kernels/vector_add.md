# vector_add

## 功能

`vector_add` 是最小的一维逐元素算子：两个向量相加，得到一个输出向量。

## 公式

```text
c[i] = a[i] + b[i]
```

## 输入输出 Shape

```text
a.shape = (N,)
b.shape = (N,)
c.shape = (N,)
```

## PyTorch Reference

代码入口：`tilelab/basic.py::vector_add_reference`。

```python
return a + b
```

Reference 的作用是给 TileLang 输出提供标准答案。

## TileLang 数据流

```text
A[idx] + B[idx] -> C[idx]
```

每个 thread 负责一个 `idx`。核心下标：

```text
idx = bx * block_size + i
```

`bx` 是 block 编号，`i` 是 block 内并行循环编号。

## 关键代码语义

- `T.Kernel(T.ceildiv(N, block_size), threads=block_size)`：启动足够多的 block 覆盖 `N` 个元素。
- `T.Parallel(block_size)`：让一个 block 内的工作并行表达。
- `if idx < N`：处理非整除尾部，避免越界。

## 边界条件

当 `N=1000`、`block_size=256`，总 thread 位置是 1024，最后 24 个位置必须跳过。

## 常见错误

- 忘记 `a.shape == b.shape`。
- CPU tensor 传给 TileLang kernel。
- 忘记 boundary guard。
- 输入不是 contiguous。

## 练习题

1. 手写 `N=1000` 时每个 block 覆盖的 index 范围。
2. 把 `block_size` 改成 128，观察需要几个 block。
3. 故意去掉 `if idx < N`，思考会发生什么。

