# Lab Index

这份索引把 starter 和 full lab 串起来。你可以把它当成“今天不知道该做什么时”的入口。

## Starter 到 Full 的桥

| Starter | 学到什么 | Full 对应内容 |
| --- | --- | --- |
| `01_vector_add_lab` | thread 下标、boundary guard | `01_basic_kernel_debug` |
| `02_gemm_lab` | `M/N/K`、tile-aligned shape | `02_gemm_tile_shape` |
| `03_tiny_model_lab` | linear/GELU/residual/logits 流程 | `04_decoder_shape_trace` |

建议先完成 starter 三个 lab，再进入 full lab。starter 少而浅，full 多而严。

## Full Lab 命令

```bash
python3 scripts/run_lab.py --list
python3 scripts/run_lab.py --lab all
python3 scripts/run_lab.py --lab reduction --run-tilelang
python3 scripts/demo_common_errors.py --case all
python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv
```

## Full Lab 清单

| Lab | 先读 | 再跑 | 产出 |
| --- | --- | --- | --- |
| basic | `labs/01_basic_kernel_debug/README.md` | `python3 scripts/run_lab.py --lab basic` | 写出一个错误如何定位 |
| gemm | `labs/02_gemm_tile_shape/README.md` | `python3 scripts/run_lab.py --lab gemm` | 写出 M/N/K 和 tile 限制 |
| reduction | `labs/03_reduction_optimization/README.md` | `python3 scripts/run_lab.py --lab reduction` | 对比 serial/parallel |
| decoder | `labs/04_decoder_shape_trace/README.md` | `python3 scripts/run_lab.py --lab decoder` | 写出 `x -> logits` shape |
| benchmark | `labs/05_benchmark_reading/README.md` | `python3 scripts/run_lab.py --lab benchmark` | 解释一条 CSV |

跑完后用 `docs/lab_answer_key.md` 校对预期输出和参考解释。

## 学习节奏

1. 读 lab README。
2. 读对应代码入口。
3. 跑 `scripts/run_lab.py`。
4. 故意触发一个错误。
5. 把“我改了什么、报错是什么、怎么修”写进报告模板。

不要急着一次跑完所有 GPU benchmark。先确认每个 shape 和错误信息都能解释。
