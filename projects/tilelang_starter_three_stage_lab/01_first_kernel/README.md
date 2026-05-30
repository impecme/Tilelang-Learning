# 01 First Kernel

## 目标

跑通第一个 TileLang kernel：`vector_add`。

你只需要先理解一件事：GPU 上会有很多 thread，每个 thread 负责一个或几个元素。

## 要读的代码

- `starter_tilelab/basic.py`
- `tests/test_basic.py`

核心公式：

```text
c[i] = a[i] + b[i]
```

核心下标：

```text
idx = bx * block_size + i
```

## 运行命令

```bash
python3 scripts/check_starter.py --stage 1
pytest tests/test_basic.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -q -m tilelang
```

## 必须回答的 3 个问题

1. `bx`、`i`、`idx` 分别代表什么？
2. `N=1000` 时为什么不能只写 `C[idx] = A[idx] + B[idx]`？
3. `vector_add_reference` 为什么要和 `vector_add_tilelang` 同时存在？

## 下一阶段入口

能解释 boundary guard 后，进入 `../02_first_matrix_op/README.md`。
