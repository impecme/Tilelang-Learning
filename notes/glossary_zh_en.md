# Glossary - AI 算子开发中英文术语对照

这份表专门记录英文工程术语和常用中文表述。读英文资料、代码和论文时，建议优先建立“英文原词 -> 中文理解 -> 代码里的样子”的三栏映射。

## 1. 调用链常用说法

英文表述：

```text
Python API -> PyTorch dispatcher -> CUDA/C++ backend -> GPU kernel -> GPU hardware
```

常用中文表述：

```text
Python 接口 -> PyTorch 分发器/调度分发层 -> CUDA/C++ 后端 -> GPU 内核/核函数 -> GPU 硬件
```

更口语一点：

```text
在 Python 里调用一个 PyTorch 函数；
PyTorch 根据算子、dtype、device 等信息选择具体实现；
真正的实现通常在 C++/CUDA 后端；
后端启动 GPU kernel；
最后由 GPU 硬件并行执行。
```

逐项解释：

| English | 常用中文 | 在本工程里的理解 |
| --- | --- | --- |
| Python API | Python 接口 | 直接调用的 Python 函数，例如 `torch.matmul`、`flash_attention_forward` |
| PyTorch dispatcher | PyTorch 分发器 / 调度分发层 | 根据算子名、dtype、device、layout 选择具体后端实现 |
| CUDA/C++ backend | CUDA/C++ 后端 | PyTorch 或扩展库底层的 C++/CUDA 实现 |
| GPU kernel | GPU 内核 / CUDA 核函数 | 在 GPU 上并行执行的函数，不是操作系统 kernel |
| GPU hardware | GPU 硬件 | SM、warp、Tensor Core、显存等实际执行单元 |

## 2. Python / PyTorch / TileLang

| English | 常用中文 | 说明 |
| --- | --- | --- |
| API | 接口 | 给用户调用的函数或类 |
| dispatcher | 分发器 / 调度器 | 根据输入类型和设备选择实现 |
| backend | 后端 | 真正执行计算的底层实现 |
| frontend | 前端 | 用户直接写的高级接口或 DSL |
| runtime | 运行时 | 程序运行期间负责调度、加载、执行的组件 |
| JIT | 即时编译 | Just-In-Time，运行时按参数编译 |
| cache | 缓存 | 保存编译结果或中间结果，避免重复工作 |
| tensor | 张量 | 多维数组，带 shape、dtype、device 等信息 |
| dtype | 数据类型 | `float16`、`bfloat16`、`float32` 等 |
| device | 设备 | CPU、CUDA GPU 等 |
| shape | 形状 | 张量每个维度的大小 |
| stride | 步幅 | 每个维度移动 1 时底层地址移动多少元素 |
| contiguous | 连续内存 / 连续布局 | 按默认布局连续存储 |
| reference | 参考实现 | 用来验证正确性的朴素实现，通常用 PyTorch 写 |
| correctness | 正确性 | 输出是否和参考实现一致 |
| benchmark | 性能测试 / 基准测试 | 测 latency、吞吐、TFLOPS 等 |
| smoke test | 冒烟测试 | 最小可运行检查，确认基本链路没坏 |

## 3. GPU 执行模型

| English | 常用中文 | 说明 |
| --- | --- | --- |
| kernel | 内核 / 核函数 | GPU 上执行的并行函数 |
| launch | 启动 kernel | 从 CPU 侧发起 GPU kernel 执行 |
| grid | 网格 | 一次 kernel launch 中的 block 组织 |
| block / program | 线程块 / 程序实例 | 一组 threads 共同计算一块输出 |
| thread | 线程 | GPU 执行的基本线程 |
| warp | 线程束 | NVIDIA GPU 上 32 个 threads 的执行组 |
| SM | 流式多处理器 | Streaming Multiprocessor，GPU 的主要执行单元 |
| occupancy | 占用率 | GPU 上活跃 warps/blocks 的程度 |
| synchronization | 同步 | 保证某些操作完成后再继续 |
| asynchronous | 异步 | CPU 发起操作后不等待 GPU 立即完成 |
| CUDA event | CUDA 事件 | 用于 GPU 时间测量和同步的对象 |

## 4. 内存层级

| English | 常用中文 | 说明 |
| --- | --- | --- |
| global memory | 全局内存 / 显存 | GPU 显存，容量大但访问慢 |
| shared memory | 共享内存 | 同一 block 内共享的高速片上内存 |
| register | 寄存器 | 线程私有的最快存储 |
| local memory | 本地内存 | 线程局部存储，可能溢出到显存 |
| fragment | 片段 / 局部 tile | TileLang 中常用于 MMA/GEMM 的局部块 |
| memory bandwidth | 内存带宽 | 单位时间读写内存的数据量 |
| memory traffic | 访存量 | 需要从内存读写的数据总量 |
| coalesced access | 合并访问 | 相邻线程访问相邻地址，提高访存效率 |
| bank conflict | bank 冲突 | shared memory 访问冲突导致变慢 |
| alignment | 对齐 | 地址满足硬件期望的边界 |

## 5. 算子与性能

| English | 常用中文 | 说明 |
| --- | --- | --- |
| operator / op | 算子 | 一个计算单元，例如 matmul、softmax |
| elementwise | 逐元素 | 每个输出元素独立计算 |
| reduction | 归约 | 多个元素合成一个或少量结果 |
| matmul | 矩阵乘法 | `A @ B` |
| GEMM | 通用矩阵乘 | General Matrix Multiply |
| tile / tiling | 分块 / 平铺 | 把大计算拆成小块 |
| pipeline | 流水线 | 让搬运和计算分阶段重叠 |
| latency | 延迟 | 单次运行耗时 |
| throughput | 吞吐 | 单位时间完成的工作量 |
| TFLOPS | 每秒万亿次浮点运算 | 衡量浮点计算吞吐 |
| arithmetic intensity | 算术强度 | 计算量与访存量之比 |
| autotuning | 自动调参 | 搜索 tile/config 找较优性能 |
| config | 配置 | 一组 kernel 参数 |
| fusion | 融合 | 把多个算子合成一个 kernel，减少中间读写 |

## 6. Attention / Transformer

| English | 常用中文 | 说明 |
| --- | --- | --- |
| token | 词元 / token | 文本被切分后的基本单位 |
| embedding | 嵌入向量 | token 对应的连续向量表示 |
| sequence length | 序列长度 | token 数量，常记作 `S` |
| hidden dimension | 隐藏维度 | 每个 token 向量的维度 |
| head | 注意力头 | multi-head attention 中的一路 attention |
| Query / Q | 查询向量 | 当前 token 想找什么信息 |
| Key / K | 键向量 | 每个 token 提供的匹配标签 |
| Value / V | 值向量 | 每个 token 真正携带的信息 |
| score | 分数 / 相似度分数 | `Q @ K^T` 得到的匹配分数 |
| softmax | 归一化指数函数 | 把分数变成权重 |
| probability / prob | 概率 / 权重 | attention 中常指 softmax 后的 `P` |
| mask | 掩码 | 屏蔽某些位置 |
| causal mask | 因果掩码 | 禁止看未来 token |
| self-attention | 自注意力 | Q/K/V 来自同一序列 |
| cross-attention | 交叉注意力 | Q 和 K/V 来自不同序列 |
| FlashAttention | FlashAttention | 分块流式 attention，避免保存完整 `(S,S)` 矩阵 |
| online softmax | 在线 softmax / 流式 softmax | 分块更新 softmax 的 `m/l/acc` 状态 |

## 7. 常见缩写

| 缩写 | 全称 | 常用中文 |
| --- | --- | --- |
| API | Application Programming Interface | 应用程序编程接口 / 接口 |
| DSL | Domain Specific Language | 领域专用语言 |
| JIT | Just-In-Time | 即时编译 |
| IR | Intermediate Representation | 中间表示 |
| TIR | Tensor Intermediate Representation | 张量中间表示 |
| CUDA | Compute Unified Device Architecture | NVIDIA CUDA 并行计算平台 |
| SM | Streaming Multiprocessor | 流式多处理器 |
| MMA | Matrix Multiply-Accumulate | 矩阵乘累加 |
| SDPA | Scaled Dot Product Attention | 缩放点积注意力 |
| FLOPs | Floating Point Operations | 浮点运算次数 |
