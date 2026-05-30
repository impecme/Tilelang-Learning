# copy

## 功能

`copy` 把输入向量逐元素复制到输出向量，是理解 global memory 读写的最小算子。

## 公式

```text
y[i] = x[i]
```

## 输入输出 Shape

```text
x.shape = (N,)
y.shape = (N,)
```

## PyTorch Reference

代码入口：`tilelab/basic.py::copy_reference`。

```python
return x.clone()
```

`clone()` 表示创建一份新 tensor，而不是只复用同一个对象。

## TileLang 数据流

```text
global X[idx] -> global Y[idx]
```

没有计算，只有读和写。

## 关键代码语义

- `T.Tensor((N,), dtype_name)` 声明一维 tensor 参数。
- `Y[idx] = X[idx]` 是一次 global memory load 和一次 global memory store。
- `idx < N` 保证尾部线程不越界。

## 边界条件

非整除长度必须有 guard。空 tensor 不作为 v1 教学目标。

## 常见错误

- 以为 copy 没有性能问题；实际上它通常受内存带宽限制。
- 用 `y = x` 当作复制，这只是让两个名字指向同一个 tensor。

## 练习题

1. 解释 `x.clone()` 和 `y = x` 的区别。
2. 思考 copy 的性能瓶颈是计算还是访存。
3. 给二维 tensor 调 `copy_tilelang`，观察它如何先 reshape 成一维。

