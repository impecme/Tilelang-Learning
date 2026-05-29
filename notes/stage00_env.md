# Stage 00 - 环境、工程结构、学习方法

## 阶段目标

这一阶段，我先建立学习工程的基本闭环：能安装、能 import、能跑测试、能跑最小 benchmark，并且知道每个目录负责什么。暂时不急着写复杂 kernel，先保证工程像一台可以反复启动的实验机器。

## 先修状态

- 会使用 Python、PyTorch、命令行的基本操作。
- 知道 CUDA kernel 大致是在 GPU 上运行的函数。
- 暂时不要求已经理解 TileLang DSL。

## 阅读

- `README.md`：环境基线、工程结构、阶段路线、验证命令。
- `notes/concepts_deep_dive.md` 第 0、8、9、13、14 节。
- `requirements.txt`：确认本工程固定 `tilelang==0.1.9`。
- `scripts/check_env.py`：看它如何检查 Python、PyTorch、CUDA、GPU、TileLang。
- `scripts/stage00_smoke.py`：看最小 smoke test 如何连接 reference 与 CUDA tensor。
- TileLang Installation Guide：重点看 pip 安装、源码构建、版本检查。

## 概念

- wheel 安装：直接使用预编译包，适合稳定学习环境。
- 源码构建：适合修改 TileLang 自身或追最新功能，但会引入 CUDA、TVM、C++ 编译链复杂度。
- driver：NVIDIA 驱动，负责让用户态程序调用 GPU。
- CUDA runtime/toolkit：编译和运行 CUDA 程序所需组件。
- PyTorch CUDA 版本：PyTorch wheel 自带或绑定的 CUDA 组件版本。
- 系统 `nvcc`：本机 CUDA 编译器版本，可能与 PyTorch CUDA 版本不同。
- GPU 架构：A100 属于 Ampere，影响 Tensor Core、shared memory、warp 行为和性能上限。
- JIT 编译：第一次调用 kernel 时编译，后续可能命中 cache。
- first-run latency：第一次运行包含编译成本，不能直接当作 kernel 性能。
- correctness test：验证结果是否正确。
- benchmark：测量速度、吞吐、瓶颈。

## 代码

- `scripts/check_env.py`：环境探测。
- `scripts/stage00_smoke.py`：最小端到端检查。
- `tests/test_vector_add.py`：普通 test 与 TileLang smoke test 如何区分。
- `pyproject.toml`：pytest marker 和 testpaths。

## 练习

1. 运行环境检查：

   ```bash
   python3 scripts/check_env.py
   ```

2. 运行最小 smoke：

   ```bash
   python3 scripts/stage00_smoke.py
   ```

3. 运行默认测试：

   ```bash
   pytest
   ```

4. 运行 TileLang 编译 smoke：

   ```bash
   RUN_TILELANG_SMOKE=1 pytest -m tilelang
   ```

5. 运行两个 benchmark smoke：

   ```bash
   python3 -m benchmarks.bench_gemm --m 128 --n 128 --k 128 --warmup 1 --repeat 2
   python3 -m benchmarks.bench_flash_attention --batch 1 --heads 1 --seq 32 --dim 64 --warmup 1 --repeat 2
   ```

6. 写 `reports/stage00_env.md`，至少包含：
   - Python 版本。
   - PyTorch 版本和 `torch.version.cuda`。
   - GPU 名称和数量。
   - TileLang 版本。
   - `pytest` 结果。
   - `RUN_TILELANG_SMOKE=1 pytest -m tilelang` 结果。
   - 是否观察到 TileLang/TVM warning，以及是否影响运行。

## 思考问题

- 为什么本工程不直接从 TileLang main 分支安装？
- 为什么第一次跑 TileLang kernel 的耗时不适合作为 benchmark 数据？
- `kernels/`、`tests/`、`benchmarks/`、`reports/` 的职责分别是什么？
- 如果 PyTorch CUDA 版本和系统 `nvcc` 版本不同，最可能影响什么？

## 验收标准

- 能稳定跑通默认测试和 TileLang smoke。
- 能解释本工程为什么优先用 PyPI wheel。
- 能说清每个顶层目录的用途。
- 完成 `reports/stage00_env.md`，并保证里面有可复现实验信息。
