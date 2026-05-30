# Lab 2：GEMM

## 目标

读懂第一个矩阵算子：`C = A @ B`，并知道 starter 版为什么要求 tile-aligned shape。

## 要读的代码

1. `starter_tilelab/advanced.py::gemm_reference`
2. `starter_tilelab/advanced.py::_require_tile_aligned`
3. `starter_tilelab/advanced.py::gemm_tilelang`

## 运行命令

```bash
python3 scripts/run_starter_lab.py --lab 2
python3 scripts/run_starter_lab.py --lab 2 --run-tilelang
```

## 动手改法

- 用 `A=(16,32), B=(32,16)` 跑一次。
- 把 `A` 改成 `(15,32)`，观察 tile-aligned 报错。
- 写下 `M/N/K` 分别来自哪个 tensor 的哪一维。

## 必须回答的 3 个问题

1. `A=(M,K), B=(K,N), C=(M,N)` 中 K 为什么必须相等？
2. starter GEMM 为什么要求 `M/N/K` 与 tile 参数对齐？
3. `T.copy -> T.gemm -> T.copy` 大概表达了什么数据流？

## 常见错误

- 把 `N` 和 `K` 混在一起。
- 用非对齐 shape 期待 TileLang 自动处理尾块。
- 忘记 fp16 GEMM 有误差容忍。

## 下一阶段入口

完成后进入 `labs/03_tiny_model_lab`，把矩阵算子放进模型流水线。
