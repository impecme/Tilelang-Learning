# axpy

## 功能

`axpy` 是经典线性代数算子，名字来自 `A * X Plus Y`。

## 公式

```text
out[i] = alpha * x[i] + y[i]
```

## 输入输出 Shape

```text
x.shape = (N,)
y.shape = (N,)
out.shape = (N,)
alpha = Python float
```

## PyTorch Reference

代码入口：`tilelab/basic.py::axpy_reference`。

```python
return float(alpha) * x + y
```

## TileLang 数据流

```text
X[idx], Y[idx], alpha -> C[idx]
```

`alpha` 是编译期参数。学习时可以理解为：每个专门的 alpha 可能生成一个专门 kernel。

## 关键代码语义

- `_compile_axpy(..., alpha)` 使用 `@lru_cache` 缓存编译结果。
- `alpha_value` 进入 TileLang factory，kernel 内直接使用。
- 和 `vector_add` 一样，尾部需要 `idx < N`。

## 边界条件

`x` 和 `y` 必须 shape 一致，dtype 一致由本项目测试约束。

## 常见错误

- 把 `alpha` 当成 tensor，但当前 v1 接口使用 Python float。
- 修改 alpha 后忘记 JIT 编译会重新发生。
- 忽略 fp16 中乘加的误差。

## 练习题

1. 把 `alpha` 分别设成 0、1、2，写出公式退化成什么。
2. 解释为什么 `alpha=0` 时结果等于 `y`。
3. 思考如果 alpha 是 tensor，接口和 kernel 会怎么变。

