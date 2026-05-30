# Stage 00.5 Exercises

这些脚本对应 `notes/stage00_5_transformer_basics.md` 的第 17 节小练习。

- `task1.py`: shape 练习，验证 `Q @ K.T`、`softmax`、`P @ V`。
- `task2.py`: batch/head 练习，验证 `(B,H,S,D)` attention shape。
- `task3.py`: 用 PyTorch 手写一次 attention，并检查 softmax 行和。
- `task4.py`: 用文字和小代码解释 `QK^T -> softmax -> P@V`。

运行方式：

```bash
python3 notes/exercise/stage00_5_transformer_basics/task1.py
python3 notes/exercise/stage00_5_transformer_basics/task2.py
python3 notes/exercise/stage00_5_transformer_basics/task3.py
python3 notes/exercise/stage00_5_transformer_basics/task4.py
```

阅读输出时重点看：

- `scores.shape` 是否符合 `Q @ K.T` 的矩阵乘法规则。
- `P.shape` 是否和 `scores.shape` 一样。
- `out.shape` 是否回到和 `V` 相同的最后一维 `D`。
- `P.sum(dim=-1)` 是否接近 1。

完成后应能用自己的话说：

```text
QK^T 计算 query 和 key 的匹配分数；
softmax 把分数变成权重；
P@V 按权重汇总 value。
```
