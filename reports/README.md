# Reports

这个目录用于保存阶段报告、benchmark 输出、debug 笔记和最终毕业项目复盘。

阶段报告命名：

- `stage00_env.md`
- `stage01_gpu_operator_model.md`
- `stage02_tilelang_dsl.md`
- `stage03_memory_debug_perf.md`
- `stage04_gemm.md`
- `stage05_autotune_operator_patterns.md`
- `stage06_attention_baseline.md`
- `stage07_flash_attention_final.md`

benchmark 数据命名：

- `stage04_gemm_a100.csv`
- `stage05_autotune_results.csv`
- `stage06_attention_baseline.csv`
- `stage07_flash_attention_bench.csv`

每份报告都应该包含足够复现实验的信息：日期、GPU、Python 版本、PyTorch 版本、TileLang 版本、shape、dtype、warmup、repeat 和命令行。
