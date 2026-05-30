# Lab 1：Vector Add

## 目标

读懂第一个 TileLang kernel：每个 thread 负责一个元素，最后一个 block 用 boundary guard 防止越界。

## 要读的代码

1. `starter_tilelab/basic.py::vector_add_reference`
2. `starter_tilelab/basic.py::_compile_vector_add`
3. `starter_tilelab/basic.py::vector_add_tilelang`

## 运行命令

```bash
python3 scripts/run_starter_lab.py --lab 1
python3 scripts/run_starter_lab.py --lab 1 --run-tilelang
```

## 动手改法

- 把 `block_size` 从 `256` 改成 `128` 调用一次。
- 分别测试 `N=1024` 和 `N=1000`。
- 故意让两个输入 shape 不一样，观察 reference 报错。

## 必须回答的 3 个问题

1. `idx = bx * block + i` 中 `bx`、`i`、`idx` 分别是什么？
2. `N=1000` 时为什么最后一个 block 有无效 thread？
3. 为什么 `if idx < N` 不能删？

## 常见错误

- 只测 `N=1024`，看不出尾块问题。
- 在 CPU tensor 上调用 TileLang kernel。
- 输入 tensor shape 不一致。

## 下一阶段入口

完成后进入 `labs/02_gemm_lab`，开始学习二维矩阵 shape。
