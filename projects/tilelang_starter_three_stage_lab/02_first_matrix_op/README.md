# 02 First Matrix Op

## 目标

跑通第一个 TileLang GEMM：`C = A @ B`。

这一阶段只学最小矩阵乘法，不学 autotune、不学复杂 tail tile、不追求高性能。

## 要读的代码

- `starter_tilelab/advanced.py`
- `tests/test_advanced.py`

核心 shape：

```text
A = (M, K)
B = (K, N)
C = (M, N)
```

核心数据流：

```text
global tile -> shared memory -> fragment -> global output
```

starter 版为了第一步稳定，`K == 32` 的路径使用 `T.gemm`。如果 K 维更大，代码会走朴素串行 fallback，保证你先能跑通和看懂 shape；完整的 pipelined GEMM 放到完整版项目继续学。

## 运行命令

```bash
python3 scripts/check_starter.py --stage 2
pytest tests/test_advanced.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -q -m tilelang
```

## 必须回答的 3 个问题

1. `M/N/K` 分别来自哪个矩阵的哪个维度？
2. 为什么 starter 版 `gemm_tilelang` 要求 shape 能被 `16x16x32` 对齐？
3. `T.copy`、`T.gemm`、`T.clear` 各自负责什么？

## 下一阶段入口

能解释 tile-aligned GEMM 后，进入 `../03_first_model_flow/README.md`。
