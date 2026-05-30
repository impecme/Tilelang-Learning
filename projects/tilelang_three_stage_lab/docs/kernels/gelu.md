# gelu

## 功能

`gelu` 是 Transformer MLP 中常见激活函数。

## 公式

```text
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

## 输入输出 Shape

```text
x.shape = 任意 contiguous tensor
y.shape = x.shape
```

TileLang v1 先把 tensor reshape 成一维处理。

## PyTorch Reference

代码入口：`tilelab/advanced.py::gelu_reference`。

```python
torch.nn.functional.gelu(x.float()).to(dtype=x.dtype)
```

## TileLang 数据流

```text
X[idx] -> erf formula -> Y[idx]
```

这是 elementwise 算子，每个输出只依赖一个输入。

## 关键代码语义

- `T.erf` 表示误差函数。
- `0.7071067811865476` 是 `1 / sqrt(2)`。
- `idx < N` 处理尾部。

## 边界条件

输入必须 CUDA 且 contiguous。输出 dtype 与输入一致。

## 常见错误

- 把 GELU 当成 ReLU。GELU 是平滑激活，不是简单截断负数。
- 忘记 reference 用 fp32 计算再 cast 回输入 dtype。
- 非 contiguous tensor 直接传入。

## 练习题

1. 对比 `relu([-1,0,1])` 和 `gelu([-1,0,1])`。
2. 找到代码里 `1/sqrt(2)` 的常量。
3. 思考为什么 MLP 常用非线性激活。

