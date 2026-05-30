# Debug Checklist

## 1. 先确认环境

```bash
python3 - <<'PY'
import torch, tilelang
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(tilelang.__version__)
PY
```

TileLang 0.1.9 在当前环境可能打印 TVM registry warning。只要 import、test、kernel 编译能通过，就先记录为已知 warning。

## 2. 输入检查

- shape 是否和文档一致。
- dtype 是否是 `float16`、`bfloat16` 或 `float32`。
- tensor 是否在 CUDA。
- tensor 是否 contiguous。
- GEMM 的 `A.shape[1]` 是否等于 `B.shape[0]`。

## 3. Correctness 排查

优先对比 PyTorch reference：

```python
actual = xxx_tilelang(...)
expected = xxx_reference(...)
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

如果不 close，先打印：

```python
diff = (actual.float() - expected.float()).abs()
idx = diff.argmax()
print(diff.max(), idx)
```

## 4. 常见错误

可以先跑错误演示脚本：

```bash
python3 scripts/demo_common_errors.py --case all
```

- 忘记 boundary guard，非整除 `N` 越界。
- 使用了 CPU tensor 调 TileLang。
- `transpose` 后没有 `.contiguous()`。
- softmax 没有减 max，导致 `exp` 溢出。
- RMSNorm 没有用 fp32 做统计量。
- causal mask 方向写反。
- GEMM 教学 tile 用在不整齐 shape 上。
- parallel reduction 的 `cols > block_n`。
- Decoder 权重 shape、dtype、device 和 config 不一致。

详细解释见 `docs/error_gallery.md`。

## 5. Benchmark 排查

- 第一次运行包含 JIT 编译，不算真实 latency。
- 使用 CUDA event，不要直接用普通 `time.time()`。
- 每条 benchmark 记录 shape、dtype、GPU、warmup、repeat。

## 6. 动手排查顺序

1. 先跑 `python3 scripts/run_lab.py --lab basic`。
2. 再跑 `python3 scripts/demo_common_errors.py --case <case>`。
3. 把报错、原因、修复方式写进报告模板的“错误定位”区域。
