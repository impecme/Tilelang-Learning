# Reports

这个目录用于保存学习记录、benchmark 表格和调试复盘。

建议命名：

- `basic_kernels_report.md`
- `advanced_ops_report.md`
- `decoder_block_report.md`
- `benchmark_YYYY_MM_DD.md`

可直接复制这些模板开始写：

- `stage01_basic_template.md`
- `stage02_advanced_template.md`
- `stage03_decoder_template.md`
- `reduction_optimization_template.md`
- `perf_optimization_template.csv`

每份报告至少记录：

- 日期。
- GPU。
- PyTorch / CUDA / TileLang 版本。
- 命令行。
- shape / dtype / warmup / repeat。
- correctness 结果。
- 我改了什么。
- 报错如何定位。
- 观察到的问题和下一步。
