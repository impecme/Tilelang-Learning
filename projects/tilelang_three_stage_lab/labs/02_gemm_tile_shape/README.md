# Lab：GEMM Tile Shape

## 目标

理解 `gemm_tilelang` 的 v1 限制：`M/N/K` 必须与 `block_m/block_n/block_k` 对齐。

## 要读的代码

1. `tilelab/advanced.py::gemm_reference`
2. `tilelab/advanced.py::_require_gemm_tile_aligned`
3. `tilelab/advanced.py::gemm_tilelang`
4. `docs/kernels/gemm.md`

## 运行命令

```bash
python3 scripts/run_lab.py --lab gemm
python3 scripts/demo_common_errors.py --case gemm_shape
```

## 动手改法

- 跑 `A=(128,128), B=(128,128)`。
- 把 `M` 改成 `96`，观察错误信息。
- 写出 `M/N/K` 和输出 `C` 的对应关系。

## 自测问题

1. 为什么 `K` 是 GEMM 的累加维？
2. 为什么 v1 不做 tail tile？
3. `T.copy` 和 `T.gemm` 分别让数据走到哪里？

## 常见错误

- 把 `block_n` 理解成输入 A 的列数。
- 忘记 `K % block_k == 0`。
- 把教学限制误认为数学限制。
