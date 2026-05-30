# Lab：Reduction Optimization

## 目标

比较串行 reduction 和 single-block parallel reduction，理解 `T.reduce_sum`、`T.reduce_max` 的角色。

## 要读的代码

1. `tilelab/basic.py::row_sum_parallel_tilelang`
2. `tilelab/advanced.py::row_softmax_parallel_tilelang`
3. `tilelab/advanced.py::rmsnorm_parallel_tilelang`
4. `docs/reduction_optimization.md`

## 运行命令

```bash
python3 scripts/run_lab.py --lab reduction
python3 scripts/run_lab.py --lab reduction --run-tilelang
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

## 动手改法

- 把 `cols` 从 `64` 改成 `128`。
- 把 `block_n` 从 `128` 改成 `256`。
- 故意设置 `cols > block_n`，观察 guard。

## 自测问题

1. 求和阶段的 neutral padding 为什么填 `0`？
2. softmax max 阶段为什么填极小值？
3. 为什么 parallel 版小 shape 不一定总更快？

## 常见错误

- 只看速度，不验证 correctness。
- 忘记 warmup/JIT 编译影响。
- 把 single-block reduction 当成通用 reduction。
