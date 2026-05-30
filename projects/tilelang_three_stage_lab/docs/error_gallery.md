# Error Gallery

这份错误图鉴帮助你把报错翻译成行动。建议配合脚本运行：

```bash
python3 scripts/demo_common_errors.py --case all
```

## CPU Tensor

现象：

```text
TileLang kernels expect CUDA tensors
```

原因：TileLang kernel 在 GPU 上运行，CPU tensor 不能直接传入。

修复：把输入放到 CUDA，或者先只跑 reference。

## Non-Contiguous Tensor

现象：

```text
TileLang kernels expect contiguous tensors
```

原因：transpose、permute 之后 tensor 的内存布局可能不连续。

修复：调用 `.contiguous()`。

## GEMM 非 Tile 对齐

现象：

```text
TileLang GEMM v1 requires tile-aligned shapes
```

原因：教学版 GEMM 没写 tail tile guard。

修复：使用 `M/N/K` 都能被 `block_m/block_n/block_k` 整除的 shape。

## Reduction 超出单 Block

现象：

```text
cols must be <= block_n
```

原因：parallel reduction v1 只做一行一个 block，不做跨 block 汇总。

修复：增大 `block_n` 到不超过 1024，或使用串行教学版。

## Decoder 权重 Shape/Dtype/Device

现象：

```text
qkv_weight must have shape ...
norm1_weight must have dtype ...
... must be on device ...
```

原因：Decoder 是多权重流水线，任意权重 shape、dtype、device 不一致都会破坏后续计算。

修复：用 `make_random_weights(config, seed, device)` 生成权重，或逐项对照 `MiniDecoderConfig`。

## 自测

1. 哪些错误不需要真正编译 TileLang 就能提前发现？
2. 为什么 Python 侧 guard 比 silent 越界更适合教学项目？
3. 如果看到 dtype 错误，第一步应该检查 config 还是 kernel？
