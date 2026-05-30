# Stage 03 Decoder Block Report

## Stage Goal

- 我是否能从 `x=(B,S,H)` 追踪到 `logits=(B,S,V)`：
- 我是否已经阅读 `docs/stages/03_decoder_block_pipeline.md`：

## Environment

- Date:
- GPU:
- Python:
- PyTorch:
- CUDA:
- TileLang:

## Commands

```bash
pytest tests/test_decoder.py
RUN_TILELANG_SMOKE=1 pytest tests/test_decoder.py -m tilelang
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --warmup 1 --repeat 1
python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --compare-optimized --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder_opt.csv
python3 scripts/run_lab.py --lab decoder
python3 scripts/demo_common_errors.py --case decoder_shape
```

## Config

| field | value |
| --- | --- |
| batch | 1 |
| seq_len | 128 |
| hidden_size | |
| num_heads | |
| head_dim | |
| ffn_hidden_size | |
| vocab_size | |
| dtype | |

## Correctness

| output | shape | tolerance | result | notes |
| --- | --- | --- | --- | --- |
| decoder block | | 3e-2 | | |
| decoder block optimized | | 3e-2 | | |
| logits | | 4e-2 | | |
| logits optimized | | 4e-2 | | |

## Data Flow Notes

- RMSNorm:
- optimized RMSNorm:
- QKV split:
- Attention:
- optimized softmax:
- MLP:
- LM Head:

## Hands-On Lab Notes

- 我改了什么 config：
- 哪一步 shape 最容易错：
- 触发过的 Decoder validation 错误：
- 我如何定位这个错误：
- 下一步只改一个什么变量：

## Stage Gate

- 我能解释 `hidden_size == num_heads * head_dim`：
- 我能画出 `QK^T -> softmax -> P@V` 的 shape：
- 我能说明 causal mask 屏蔽未来 token：
- 我能指出每个 TileLang kernel 在流水线中的位置：
- 我能说明 optimized 入口只替换了哪些 reduction 算子：
- 本项目 v1 是否完成：

## Benchmark Results

| path | latency ms | notes |
| --- | --- | --- |
| reference | | |
| tilelang | | |
| tilelang optimized reductions | | |

## Open Questions

-
