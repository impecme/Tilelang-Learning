# Reduction Optimization Report

## 实验信息

| 项目 | 内容 |
| --- | --- |
| 日期 | |
| GPU | |
| dtype | |
| rows | |
| cols | |
| block_n | |
| CSV 路径 | |

## 运行命令

```bash
python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv
```

## Correctness

| 算子 | tolerance | 是否通过 | 备注 |
| --- | --- | --- | --- |
| row_sum_parallel | | | |
| row_softmax_parallel | | | |
| rmsnorm_parallel | | | |
| mini_inference_tilelang_optimized | | | |

## Latency

| 算子 | torch ms | serial TileLang ms | parallel TileLang ms | speedup(serial/parallel) | 观察 |
| --- | --- | --- | --- | --- | --- |
| row_sum | | | | | |
| row_softmax | | | | | |
| rmsnorm | | | | | |

## Decoder Before/After

| 路径 | latency ms | 说明 |
| --- | --- | --- |
| PyTorch reference | | |
| tilelang_decoder_block | | 串行 reduction |
| tilelang_decoder_block_optimized_reductions | | 并行 reduction |

## 结论

- 哪个算子收益最大：
- 哪个算子收益最小：
- 可能原因：
- 下一轮只改一个变量：
