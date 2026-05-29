# Stage 07 - FlashAttention Forward Capstone

## 阶段目标

这一阶段，我完成毕业项目：保持 public API 不变，把当前 PyTorch online softmax reference 逐步替换为 non-causal TileLang FlashAttention forward。目标是 correctness first，然后再做性能分析。

## 先修状态

- 已完成 Stage 06。
- 能解释 online softmax 的 `m/l/acc`。
- 能跑 attention benchmark。
- 理解 GEMM tile、shared memory、fragment、pipeline。

## 阅读

- TileLang examples 中 attention/flash attention 相关实现。
- `notes/concepts_deep_dive.md` 第 10、11、12、13、14 节。
- `kernels/flash_attention.py`。
- `kernels/reference.py`。
- `tests/test_flash_attention_reference.py`。
- `benchmarks/bench_flash_attention.py`。
- Stage 04 GEMM 报告。
- Stage 06 online softmax 推导。

## 概念

- Q block：一次处理的一段 query。
- K/V block：流式遍历的一段 key/value。
- head dimension specialization：优先支持 `D=64` 和 `D=128`。
- non-causal first：先做无 mask 版本，降低复杂度。
- causal optional：后续再加入 `j <= i` mask。
- fp32 `m/l/acc`：提升 softmax 和累加稳定性。
- output dtype：输出 dtype 跟输入一致。
- boundary guard：处理 `S` 不是 block size 整数倍。
- tail block：最后一个不足 block size 的 tile。
- PyTorch SDPA：优化 baseline，不要求第一版超过它。
- materialized naive attention：必须超过的基础 baseline。

## 代码

- `kernels/flash_attention.py`
  - public API，签名不能改。
  - 当前实现调用 `online_attention_forward`。
- `kernels/reference.py`
  - correctness oracle。
- `tests/test_flash_attention_reference.py`
  - 必须保持通过的测试语义。
- `benchmarks/bench_flash_attention.py`
  - 最终性能对比入口。

## 实现要求

- public API 保持不变：

  ```python
  flash_attention_forward(q, k, v, causal=False, sm_scale=None)
  ```

- 输入 shape：
  - `q/k/v`: `(B, H, S, D)`。
  - 第一版可以假设 `q/k/v` sequence length 相同。

- dtype：
  - 支持 `float16`。
  - 支持 `bfloat16` 作为目标。
  - 内部 `m/l/acc` 使用 `float32`。

- shape：
  - 必须支持 `D=64`。
  - 必须支持 `D=128`。
  - 必须支持非整除 `S`。

- mask：
  - 必做 non-causal。
  - causal 可选，不阻塞毕业验收。

## 实现路径

1. 保留当前 PyTorch reference 作为 fallback。
2. 新增 TileLang kernel 编译函数，不改变 public API。
3. 先只处理小 shape：`(1,1,128,64)`。
4. 再处理非整除 `S`：例如 `(1,1,129,64)`。
5. 再扩展 batch/head：`(2,8,512,64)`。
6. 再扩展 `D=128`。
7. 最后接入 benchmark，和 naive attention、PyTorch SDPA 对比。

## 练习

1. Correctness:
   - fp16 non-causal。
   - bf16 non-causal。
   - `D=64`。
   - `D=128`。
   - 非整除 `S`。

2. Benchmark:
   - `(1,1,128,64)`。
   - `(2,8,512,64)`.
   - `(1,16,1024,128)`.
   - 至少记录 naive、PyTorch SDPA、TileLang 三列。

3. 误差分析：
   - 记录 max abs diff。
   - 记录 max relative diff。
   - 对最大误差 index 打印 actual/expected。

4. 性能分析：
   - 记录 block_M/block_N/D/threads/num_stages。
   - 记录 latency。
   - 说明是否快于 materialized naive attention。
   - 说明与 PyTorch SDPA 差距的可能来源。

## 报告模板

建议写 `reports/stage07_flash_attention_final.md`：

```markdown
# Stage 07 FlashAttention Final Report

## API

## Supported Shapes and Dtypes

## Algorithm

## Correctness Results

| shape | dtype | causal | max abs diff | max rel diff | pass |
| --- | --- | --- | --- | --- | --- |

## Benchmark Results

| shape | dtype | naive ms | sdpa ms | tilelang ms | notes |
| --- | --- | --- | --- | --- | --- |

## Known Limitations

## Gap vs PyTorch SDPA

## Next Steps
```

## 思考问题

- 为什么第一版先做 non-causal？
- 为什么 `D=64/128` 适合作为第一阶段目标？
- `m/l/acc` 分别应该放在哪里？
- 如果输出正确但性能不如 naive attention，可能有哪些原因？
- 如果和 PyTorch SDPA 差距很大，下一步应该优先查访存、occupancy 还是 tile config？

## 验收标准

- public API 不变。
- correctness 覆盖 fp16/bf16、D=64/128、非整除 S。
- non-causal TileLang 版本快于 materialized naive attention。
- 能解释与 PyTorch SDPA 的主要差距。
- 完成 `reports/stage07_flash_attention_final.md`。
