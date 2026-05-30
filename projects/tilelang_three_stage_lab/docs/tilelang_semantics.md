# TileLang Semantics

这份文档解释本项目反复出现的 TileLang 语义。

## `@tilelang.jit`

`@tilelang.jit` 修饰的是 kernel factory。factory 接收 shape、tile config、dtype 等编译期参数，返回真正的 `T.prim_func`。

学习时可以这样理解：

```text
Python factory 参数 -> TileLang 编译 -> GPU kernel -> PyTorch tensor 调用
```

## `T.prim_func`

`T.prim_func` 里的代码不是普通 Python 循环。它会被 TileLang/TIR 解释并编译成 GPU kernel。

所以在 `T.prim_func` 里写：

```python
for i in T.Parallel(block_size):
    ...
```

不是 Python 逐个执行，而是在表达 GPU block 内的并行工作。

## `T.Kernel`

```python
with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
    ...
```

- `grid_x/grid_y`：有多少个 program/block。
- `threads`：每个 block 中有多少 threads。
- `bx/by`：当前 block 坐标。

## `T.Parallel`

`T.Parallel` 用来表达 block 内可以并行的循环。最常见的 elementwise 写法：

```python
with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
    for i in T.Parallel(block_size):
        idx = bx * block_size + i
        if idx < N:
            Y[idx] = X[idx]
```

## `T.serial`

`T.serial` 表达顺序循环。它常用于 reduction 的入门版本：

```python
acc = 0
for col in T.serial(N):
    acc += X[row, col]
```

v1 中 `row_sum`、`row_softmax`、`rmsnorm` 使用 serial reduction，目的是把公式讲清楚。

## `T.reduce_sum`、`T.reduce_max`

parallel reduction 教学版使用 `T.reduce_sum` 和 `T.reduce_max` 把一个 fragment 中的多个值汇总成一个值：

```python
values = T.alloc_fragment((block_n,), "float32")
total = T.alloc_fragment((1,), "float32")
T.reduce_sum(values, total, dim=0, clear=True)
```

本项目只用它们做单 block reduction：一行数据必须能放进 `block_n` 个 thread 覆盖的范围内。

## `T.copy`、`T.gemm`

GEMM 使用：

- `T.copy`：把 global memory tile 搬到 shared memory。
- `T.gemm`：调用矩阵乘累加路径，更新 fragment accumulator。

如果你看到 `T.gemm(A_shared, B_shared, C_local)`，就把它理解为：

```text
C_local += A_shared @ B_shared
```

## `out_idx`

`@tilelang.jit(out_idx=[-1])` 表示最后一个参数是输出，TileLang adapter 可以帮忙处理输出 tensor。本项目 GEMM 使用这个模式。
