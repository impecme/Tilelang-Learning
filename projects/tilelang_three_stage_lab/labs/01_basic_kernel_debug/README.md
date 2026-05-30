# Lab：Basic Kernel Debug

## 目标

通过 `vector_add` 和 `row_sum` 学会读报错：shape、device、contiguous、boundary guard 分别是什么问题。

## 要读的代码

1. `tilelab/basic.py::vector_add_reference`
2. `tilelab/basic.py::vector_add_tilelang`
3. `tilelab/basic.py::row_sum_tilelang`
4. `tilelab/basic.py::row_sum_parallel_tilelang`

## 运行命令

```bash
python3 scripts/run_lab.py --lab basic
python3 scripts/demo_common_errors.py --case cpu_tensor
python3 scripts/demo_common_errors.py --case non_contiguous
```

## 动手改法

- 把 vector_add 的输入长度改成 `1000`，解释尾块。
- 故意传入不同 shape 的 tensor，确认 reference 先报错。
- 对比 `row_sum_tilelang` 和 `row_sum_parallel_tilelang` 的限制。

## 自测问题

1. 为什么 CPU tensor 不能直接传给 TileLang kernel？
2. non-contiguous tensor 为什么危险？
3. `row_sum_parallel_tilelang` 为什么要求 `cols <= block_n`？

## 常见错误

- 只看异常类型，不看异常信息。
- 没先跑 reference，就直接跑 TileLang。
- 以为所有 shape 都能自动处理尾块。
