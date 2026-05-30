# 03 Decoder Block Pipeline

这一层把前两层的算子串成一个迷你 Decoder Block。它不是完整 LLM，而是一个足够真实的推理子模块，用来学习 shape 流、权重流和算子编排。

详细任务见 `../docs/stages/03_decoder_block_pipeline.md`，三阶段总览见 `../docs/stage_map.md`。

## 阶段定位

- 阶段：大型综合阶段。
- 学习对象：单 Decoder Block + LM Head logits。
- 核心代码：`../tilelab/decoder.py`。
- 测试文件：`../tests/test_decoder.py`。
- 报告模板：`../reports/stage03_decoder_template.md`。
- 动手 lab：`../labs/04_decoder_shape_trace/README.md`。

## 细化学习目标

- 会解释 `MiniDecoderConfig` 每个字段如何影响 shape。
- 会说明 `hidden_size == num_heads * head_dim` 为什么必须成立。
- 会从 `x=(B,S,H)` 推导 QKV、attention、MLP、logits 的 shape。
- 会指出哪些步骤调用了 Stage 01/02 的 TileLang 算子。
- 会解释两次 RMSNorm、两次 residual add、LM Head 的位置。
- 会对比 optimized 入口中 RMSNorm/softmax 的并行 reduction 替换。
- 会让 TileLang 路径输出与 PyTorch reference 对齐。

## 流水线

```text
x
 -> RMSNorm
 -> QKV Linear
 -> Causal Attention
 -> Output Linear
 -> Residual Add
 -> RMSNorm
 -> MLP Up Linear + GELU
 -> MLP Down Linear
 -> Residual Add
 -> LM Head
 -> logits
```

## 默认 shape

`MiniDecoderConfig` 默认：

```text
B = 1
S = 128
hidden_size = 256
num_heads = 4
head_dim = 64
ffn_hidden_size = 1024
vocab_size = 4096
dtype = float16
```

输入 `x.shape = (B, S, hidden_size)`。

输出：

- `decoder_block_reference/tilelang`：`(B, S, hidden_size)`。
- `mini_inference_reference/tilelang`：`(B, S, vocab_size)` logits。
- `decoder_block_tilelang_optimized`：同 shape，用并行 reduction 版本替换 RMSNorm 和 softmax。

## Attention Shape

QKV linear 之后：

```text
qkv.shape = (B, S, 3*hidden)
q.shape = (B, H, S, D)
k.shape = (B, H, S, D)
v.shape = (B, H, S, D)
scores.shape = (B, H, S, S)
probs.shape = (B, H, S, S)
attention_out.shape = (B, S, hidden)
```

`scores = Q @ K^T / sqrt(D)`。如果 `causal=True`，第 i 个 query 只能看 `j <= i` 的 key。

## TileLang 覆盖

v1 尽量使用 TileLang：

- RMSNorm：`rmsnorm_tilelang`
- optimized RMSNorm：`rmsnorm_parallel_tilelang`
- QKV/O/MLP/LM Head linear：`gemm_tilelang`
- causal mask：`causal_mask_tilelang`
- softmax：`row_softmax_tilelang`
- optimized softmax：`row_softmax_parallel_tilelang`
- residual add：`vector_add_tilelang`
- GELU：`linear_bias_gelu_tilelang`

Python 负责 reshape、split head、循环 batch/head。这是学习型实现，方便观察每一步 tensor shape。

## 逐步任务

1. 读 `../docs/tensor_shapes.md`，写出默认 config 的关键 shape。
2. 读 `MiniDecoderConfig` 和 `MiniDecoderWeights`，理解配置与权重 shape 的关系。
3. 读 reference 路径，确认 PyTorch 标准答案的计算顺序。
4. 读 TileLang 路径，标出每个被复用的 Stage 01/02 kernel。
5. 读 `../docs/optimization_log.md`，标出哪些 PyTorch 小操作被替换成 TileLang utility kernel。
6. 读 `../docs/reduction_optimization.md`，说明 optimized 入口替换了哪些串行 reduction。
7. 运行 decoder correctness 和 TileLang smoke。
8. 把 `x -> logits` 的完整 shape 流和 benchmark CSV 写入综合报告。

## 运行

```bash
python3 scripts/check_project.py --stage decoder
python3 scripts/run_lab.py --lab decoder
pytest tests/test_decoder.py -q
RUN_TILELANG_SMOKE=1 pytest tests/test_decoder.py -q -m tilelang
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --warmup 1 --repeat 1
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --compare-optimized --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv
```

## 性能学习入口

Decoder TileLang 路径现在尽量避免 PyTorch 小操作混入：

- GEMM 后 bias add 使用 `linear_bias_tilelang`。
- attention score 的 scale 使用 `scale_tilelang`。
- causal 路径把 scale 和 mask 合成 `scale_causal_mask_tilelang`。
- optimized 入口把 RMSNorm 和 attention softmax 换成 parallel reduction 版本。

这仍然是教学版，不是最终高性能 fused kernel。重点是学会看数据流、看 CSV、写解释。

## 常见错误

- `hidden_size != num_heads * head_dim`。
- `q/k/v` 的 shape 顺序混成 `(B,S,H,D)` 或 `(B,H,S,D)`。
- 忘记 `k.transpose(-2, -1)`。
- causal mask 用错方向，把过去屏蔽掉。
- GEMM tile 版本遇到非整齐 shape 时需要先固定教学 shape。

## 验收标准

- 能解释 Decoder Block 中两次 RMSNorm 的位置。
- 能画出 `QK^T -> softmax -> P@V` 的 shape。
- 能说明 residual add 为什么要保留原输入。
- 能运行 reference smoke。
- 能运行 TileLang tiny smoke，并知道它是 correctness-first，不代表最终性能。
- 能让 `mini_inference_tilelang` 的 logits 与 PyTorch reference close。
- 能让 `mini_inference_tilelang_optimized` 的 logits 与 PyTorch reference close。
- 能解释优化后的 TileLang 路径少了哪些 PyTorch 小操作。
- 能运行 decoder lab，并解释一个权重 shape/dtype/device 错误。

## 项目完成条件

完成 `../reports/stage03_decoder_template.md` 的学习记录，并且能从 `x=(B,S,H)` 一路解释到 `logits=(B,S,V)`，就完成了本项目 v1 的核心学习目标。
