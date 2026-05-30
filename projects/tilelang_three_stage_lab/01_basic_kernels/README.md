# 01 Basic Kernels

这一层是 TileLang 学习的地基。目标不是性能，而是让你真正理解一个 GPU kernel 如何把大量元素分给 block/thread，并且知道为什么边界保护、dtype、device、contiguous 检查会影响正确性。

详细任务见 `../docs/stages/01_basic_kernels.md`，三阶段总览见 `../docs/stage_map.md`。

## 阶段定位

- 阶段：基础阶段。
- 学习对象：最小 elementwise kernel 和入门 reduction。
- 核心代码：`../tilelab/basic.py`。
- 测试文件：`../tests/test_basic.py`。
- 报告模板：`../reports/stage01_basic_template.md`。
- 动手 lab：`../labs/01_basic_kernel_debug/README.md`。

## 细化学习目标

- 会解释 `@tilelang.jit`、`T.Kernel`、`T.Parallel` 的基本作用。
- 会推导一维全局下标：`idx = bx * block_size + i`。
- 会说明 `T.ceildiv` 为什么会产生尾块。
- 会解释 `if idx < N` 如何避免越界。
- 会用 PyTorch reference 验证 TileLang 输出。
- 会区分 elementwise 和 reduction。
- 会读懂 `row_sum_parallel_tilelang` 的单 block 并行 reduction 限制。

## 算子清单

| 算子 | 数学公式 | 输入 | 输出 | 重点语义 |
| --- | --- | --- | --- | --- |
| `copy` | `y[i] = x[i]` | `x=(N,)` | `y=(N,)` | global memory 读写 |
| `vector_add` | `c[i] = a[i] + b[i]` | `a,b=(N,)` | `c=(N,)` | block/thread 分工 |
| `axpy` | `out[i] = alpha*x[i] + y[i]` | `x,y=(N,)` | `out=(N,)` | 编译期常量参数 |
| `row_sum` | `y[m] = sum_n x[m,n]` | `x=(M,N)` | `y=(M,)` | reduction 入门 |
| `row_sum_parallel` | 同上 | `x=(M,N)` | `y=(M,)` | `T.reduce_sum` 入门 |

## TileLang 语义

`T.Kernel(T.ceildiv(N, block_size), threads=block_size)` 表示启动若干个 GPU block。每个 block 有 `block_size` 个 threads。

`T.Parallel(block_size)` 表示 block 内的并行循环。可以把循环变量 `i` 理解成“当前 thread 负责的局部编号”。

一维 elementwise kernel 的全局下标通常是：

```text
idx = bx * block_size + i
```

- `bx`：当前 block 在 grid 中的位置。
- `i`：当前 block 内的 thread 位置。
- `idx`：当前 thread 负责的全局元素。

## Boundary Guard

当 `N=1000`、`block_size=256` 时，`ceildiv(1000,256)=4`，总共有 `4*256=1024` 个 thread 位置。最后 24 个位置没有真实元素，所以 TileLang kernel 必须写：

```python
if idx < N:
    C[idx] = A[idx] + B[idx]
```

没有 boundary guard，kernel 可能读写越界。基础阶段必须同时测试 `N=1024` 和 `N=1000`，因为只测整除 shape 看不出尾块问题。

## 逐步任务

1. 读 `../docs/kernels/vector_add.md`，手算 `N=1000, block_size=256` 的 block 覆盖范围。
2. 读 `copy_tilelang`，指出输入读和输出写的位置。
3. 读 `axpy_tilelang`，解释 `alpha` 是如何进入 kernel 的。
4. 读 `row_sum_tilelang`，说明一行输入如何合成一个输出。
5. 读 `../docs/reduction_optimization.md`，说明 `row_sum_parallel_tilelang` 为什么要求 `cols <= block_n`。
6. 运行基础测试和 TileLang smoke。
7. 把结果记录到基础阶段报告。

## 运行

```bash
python3 scripts/check_project.py --stage basic
python3 scripts/run_lab.py --lab basic
pytest tests/test_basic.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -q -m tilelang
python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

## 验收标准

- 能解释 `idx = bx * block_size + i`。
- 能说明为什么 `N=1000` 需要 boundary guard。
- 能说出 elementwise 和 reduction 的区别。
- 能解释 serial row_sum 和 parallel row_sum 的区别。
- 能用 PyTorch reference 验证 TileLang 输出。
- 能根据报错判断 shape、dtype、device、contiguous 哪一类出了问题。
- 能运行 basic lab，并解释至少一个常见错误。

## 常见错误

- 只运行 PyTorch reference，没有打开 TileLang smoke。
- 用 CPU tensor 调用 TileLang kernel。
- 输入 tensor 非 contiguous。
- 忘记测试非整除长度。
- 把 `row_sum` 当成每个输出只依赖一个输入的 elementwise。

## 进入下一阶段条件

完成 `../reports/stage01_basic_template.md` 的学习记录，并且能独立解释 boundary guard 后，再进入 `02_advanced_ops`。
