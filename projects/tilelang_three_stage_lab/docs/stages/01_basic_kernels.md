# Stage 01：Basic Kernels

基础阶段的目标不是写出最快 kernel，而是建立“GPU 上谁负责哪个元素”的直觉。你需要能读懂代码、改小参数、跑测试，并能解释为什么输出是正确的。

## 阶段定位

- 难度：入门。
- 核心文件：`tilelab/basic.py`。
- 对应测试：`tests/test_basic.py`。
- 对应 benchmark：`benchmarks/bench_basic.py`。
- 对应 lab：`labs/01_basic_kernel_debug/README.md`。
- 学习报告：`reports/stage01_basic_template.md`。

## 细化学习目标

- 能说出 `xxx_reference` 和 `xxx_tilelang` 的职责差别。
- 能解释 `T.Kernel(T.ceildiv(N, block_size), threads=block_size)` 启动了多少 block 和 thread。
- 能把 `bx`、`i`、`idx` 分别对应到 block id、thread-local id、global element id。
- 能说明 `N=1000` 时为什么会出现尾块。
- 能解释 `.is_cuda`、`.is_contiguous()`、`dtype` 检查为什么放在 Python 侧。
- 能区分 `vector_add` 这类 elementwise 和 `row_sum` 这类 reduction。
- 能说明 `row_sum_parallel_tilelang` 为什么是一行一个 block 的入门并行 reduction。

## 必会概念

| 概念 | 小白解释 |
| --- | --- |
| `@tilelang.jit` | 把 Python 描述的 kernel 编译成可运行的 GPU kernel |
| `T.Kernel` | 定义 GPU grid 和每个 block 的线程数 |
| `T.Parallel` | 表示 block 内多个 thread 并行处理局部循环 |
| `T.ceildiv` | 向上取整，保证所有元素都有 thread 位置覆盖 |
| boundary guard | 防止多出来的 thread 读写越界 |
| contiguous | tensor 在内存中连续，kernel 才能按简单下标访问 |
| neutral padding | 尾部无效位置填不会影响结果的值，例如求和填 0 |

## 逐步任务

1. 阅读 `docs/kernels/vector_add.md`，在纸上写出 `N=1000, block_size=256` 时每个 block 覆盖的范围。
2. 阅读 `copy_tilelang`，找出 global memory 读和写各发生在哪一行。
3. 阅读 `axpy_tilelang`，说明 `alpha` 为什么不需要是一个 tensor。
4. 阅读 `row_sum_tilelang`，写出一行输入如何变成一个输出。
5. 阅读 `docs/reduction_optimization.md` 的 `row_sum` 部分，理解串行版和 parallel 版的区别。
6. 运行 `python3 scripts/run_lab.py --lab basic`，确认你能解释错误和 shape。
7. 运行基础测试，确认 reference 和 TileLang 输出一致。
8. 把 benchmark 结果写进报告模板，并备注第一次编译时间不能当成稳定 latency。

## 必须运行

```bash
python3 scripts/check_project.py --stage basic
python3 scripts/run_lab.py --lab basic
pytest tests/test_basic.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -q -m tilelang
python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

## 实验记录要求

- 至少记录 `N=1024` 和 `N=1000` 两个 shape。
- 对 `vector_add`、`copy`、`axpy`、`row_sum` 分别写一句输入输出语义。
- 记录 `row_sum` 串行版和 parallel 版的一次 latency 对比。
- 记录是否跑过 TileLang smoke，以及是否遇到 CUDA/dtype/contiguous 报错。

## 验收问题

- `idx = bx * block_size + i` 中三个变量分别代表什么？
- 为什么 `ceildiv` 后可能会多出 thread 位置？
- boundary guard 写错会导致什么风险？
- 为什么 `row_sum` 的输出数量比输入元素数量少？
- `row_sum_parallel_tilelang` 中 `cols > block_n` 为什么会被拒绝？

## 常见卡点

- 只跑 PyTorch reference，没有设置 `RUN_TILELANG_SMOKE=1`。
- 在 CPU tensor 上调用 TileLang kernel。
- 输入是 transpose 后的非 contiguous tensor。
- 改了 `block_size` 但忘记重新理解尾块。

## 进入下一阶段条件

当你能独立解释边界保护，并且能根据测试报错判断是 shape、dtype、device 还是 contiguous 问题时，再进入 Stage 02。
