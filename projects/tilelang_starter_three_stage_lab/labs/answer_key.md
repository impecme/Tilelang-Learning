# Starter Lab Answer Key

这份参考答案不是为了替代动手练习，而是让你跑完后能确认自己看到的现象是否正常。

## Lab 1 - Vector Add

推荐命令：

```bash
python3 scripts/run_starter_lab.py --lab 1
```

预期观察：

- reference 会创建两个一维 tensor，输出 `a + b`。
- `N=1000` 这类非整除长度仍然应该正确，因为 kernel 里有 boundary guard。
- `idx = block_id * block_size + thread_id` 是每个线程负责的元素下标。

自测问题参考：

- `T.Kernel` 描述 grid 和 threads。
- `T.Parallel` 描述一个 block 内的并行线程循环。
- `if idx < n` 是为了保护最后一个不满 block 的 tail。

常见错误：

- 忘记 boundary guard，`N=1000` 时容易越界。
- CPU tensor 传给 TileLang kernel，会被 device 检查拦住。

## Lab 2 - GEMM

推荐命令：

```bash
python3 scripts/run_starter_lab.py --lab 2
```

预期观察：

- `A.shape = (M, K)`，`B.shape = (K, N)`，输出 `C.shape = (M, N)`。
- starter v1 只允许 `M/N/K` 和 block 对齐；非对齐 shape 应该抛 `ValueError`。
- shared memory 用来把 global memory 的 tile 搬近一点，fragment 用来表示每个 program 内部的计算片段。

自测问题参考：

- `block_m/block_n/block_k` 决定一个 program 负责的输出 tile 和 K 方向分块。
- `T.copy` 是显式数据搬运。
- `T.gemm` 是 tile 内矩阵乘加。

常见错误：

- 期待非 tile-aligned shape 自动处理 tail。starter 版故意不做，方便先看懂主路径。
- 把 `A @ B` 的 `K` 维写反。

## Lab 3 - Tiny Model Flow

推荐命令：

```bash
python3 scripts/run_starter_lab.py --lab 3
```

预期 shape：

```text
x:        (B, S, hidden)
linear1:  (B*S, ffn_hidden)
GELU:     (B*S, ffn_hidden)
linear2:  (B*S, hidden)
residual: (B, S, hidden)
lm_head:  (B*S, vocab)
logits:   (B, S, vocab)
```

自测问题参考：

- 这个模型流不是完整 Decoder Block，因为没有 attention、RMSNorm 和 KV cache。
- residual 要求两边 shape 一样。
- logits 最后一维是 vocab size，表示每个 token 对词表的分数。

常见错误：

- 忘记 `.contiguous()`，导致后续 kernel 的 contiguous 检查失败。
- 把 `hidden_size` 和 `ffn_hidden_size` 混用。

