# Concepts Deep Dive - TileLang AI 算子开发细讲版

这份文档是 `notes/concepts.md` 的细讲版。`concepts.md` 用来快速建立词汇表，本文件用来把每个概念讲透一点：它是什么、为什么重要、在代码里长什么样、常见错误是什么、应该如何验证。

建议读法：

1. 第一次学习时，按 Stage 顺序读对应章节。
2. 写 kernel 卡住时，回到“代码对应关系”和“常见错误”部分查。
3. 做阶段复盘时，用每节最后的自检问题检查自己是否真的理解。

## 0. 从 PyTorch 算子到 GPU Kernel

在 PyTorch 中，通常会写：

```python
y = torch.matmul(x, w)
z = torch.nn.functional.softmax(scores, dim=-1)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

这些函数看起来像普通 Python 调用，但真正耗时的部分通常不是 Python 在算，而是底层 GPU kernel 在算。可以把调用链粗略理解成：

```text
Python API -> PyTorch dispatcher -> CUDA/C++ backend -> GPU kernel -> GPU hardware
```

常用中文表述：

```text
Python 接口 -> PyTorch 分发器/调度分发层 -> CUDA/C++ 后端 -> GPU 内核/核函数 -> GPU 硬件
```

逐项对应：

| English | 常用中文 | 说明 |
| --- | --- | --- |
| Python API | Python 接口 | 直接调用的 Python 函数，例如 `torch.matmul` |
| PyTorch dispatcher | PyTorch 分发器 / 调度分发层 | 根据算子、dtype、device、layout 选择具体实现 |
| CUDA/C++ backend | CUDA/C++ 后端 | PyTorch 或扩展库底层的 C++/CUDA 实现 |
| GPU kernel | GPU 内核 / CUDA 核函数 | 在 GPU 上并行执行的函数 |
| GPU hardware | GPU 硬件 | SM、warp、Tensor Core、显存等实际执行单元 |

完整术语表见 `notes/glossary_zh_en.md`。

TileLang 学习的重点，是从“只调用 PyTorch 算子”推进到“能自己描述底层 kernel 如何计算”。也就是说，接下来需要开始关心：

- 每个 GPU block 负责哪一块输出。
- 每个 thread 做什么。
- 输入数据如何从 global memory 读进来。
- 中间结果放在 shared memory、fragment 还是 register。
- 输出何时写回 global memory。
- 如何证明结果正确。
- 如何测量性能。

一个 AI 算子不是单纯的数学公式，而是“数学公式 + 数据布局 + 并行策略 + 内存策略 + 数值策略 + 测试策略”的组合。

## 1. AI 算子到底是什么

AI 算子是模型中的一个计算单元。它可以很小，例如：

```text
out[i] = a[i] + b[i]
```

也可以很大，例如 attention：

```text
scores = Q @ K^T * scale
probs = softmax(scores)
out = probs @ V
```

写算子时要把问题拆成六层：

| 层次 | 需要回答的问题 | 例子 |
| --- | --- | --- |
| 数学层 | 要算什么 | `C = A @ B` |
| Shape 层 | 输入输出是什么形状 | `A[M,K]`, `B[K,N]`, `C[M,N]` |
| Layout 层 | 数据怎么放在内存里 | contiguous, stride |
| 并行层 | 谁负责算哪一块 | block 负责 `128x128` tile |
| 内存层 | 数据在哪里复用 | A/B tile 放 shared |
| 验证层 | 如何知道对不对、快不快 | PyTorch reference + benchmark |

常见误区：

- 只看数学公式，不看 shape 和 layout。
- 只跑一个 shape，不测边界 shape。
- 一开始就追性能，没有 reference test。
- benchmark 把首次 JIT 编译时间也算进去。

自检问题：

- 能否用一句话说清这个算子输入、输出和数学公式？
- 能否写出 PyTorch reference？
- 能否列出至少 3 个会导致结果错误的边界条件？

## 2. Tensor、Shape、Stride、Layout

Tensor 可以理解成“带元信息的多维数组”。关键元信息包括：

- `shape`：逻辑维度。
- `dtype`：元素类型。
- `device`：CPU 或 CUDA。
- `stride`：每个维度移动 1 时，底层存储移动多少个元素。
- `layout`：数据排列方式。

以 attention 的 `q` 为例：

```text
q.shape = (B, H, S, D)
```

含义：

- `B`：batch size。
- `H`：num heads。
- `S`：sequence length。
- `D`：head dimension。

如果 `q` 是 contiguous 的 `(B, H, S, D)`，通常最后一维 `D` 是连续的。一个元素 `q[b, h, s, d]` 的线性 offset 可以理解为：

```text
offset = ((b * H + h) * S + s) * D + d
```

这件事很重要，因为 GPU 喜欢连续访问。比如一组相邻 threads 访问：

```text
q[b, h, s, d + 0]
q[b, h, s, d + 1]
q[b, h, s, d + 2]
...
```

通常比它们跳着访问更高效。

在 TileLang kernel 里，通常先假设输入是 contiguous，然后在 Python 包装层检查：

```python
if not x.is_contiguous():
    raise ValueError("expected contiguous tensor")
```

常见错误：

- 把 `(B, S, H, D)` 当成 `(B, H, S, D)`。
- 忘记 `.contiguous()`，导致 stride 与 kernel 假设不一致。
- 输出 shape 正确，但 layout 假设错，结果整体错位。

自检问题：

- 给定一个 4D contiguous tensor，能否写出 offset 公式？
- 当前 kernel 是否要求输入 contiguous？
- 如果输入是 transpose 后的 tensor，会发生什么？

## 3. GPU 执行层级：Grid、Block、Thread、Warp

GPU kernel 启动时会生成很多 block，每个 block 里有很多 threads。可以粗略理解为：

```text
grid
  block 0
    thread 0
    thread 1
    ...
  block 1
    thread 0
    thread 1
    ...
```

NVIDIA GPU 中，warp 是 32 个 threads 的执行组。代码里写的是 thread 级逻辑，但硬件经常以 warp 为单位调度。

TileLang 中常见写法：

```python
with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
    for i in T.Parallel(block_size):
        idx = bx * block_size + i
        C[idx] = A[idx] + B[idx]
```

逐行理解：

- `T.ceildiv(N, block_size)`：需要多少个 block 才能覆盖 `N` 个元素。
- `threads=block_size`：每个 block 有多少 threads。
- `bx`：当前 block 的编号。
- `T.Parallel(block_size)`：block 内并行循环。
- `idx`：当前 thread 负责的全局元素 index。

如果 `N=1024`，`block_size=256`：

```text
block 0 -> idx 0..255
block 1 -> idx 256..511
block 2 -> idx 512..767
block 3 -> idx 768..1023
```

如果 `N=1000`，最后一个 block 会覆盖 `768..1023`，其中 `1000..1023` 越界。因此通用 kernel 需要 boundary guard：

```text
if idx < N:
    C[idx] = A[idx] + B[idx]
```

常见错误：

- 只测试 `N` 能整除 block size 的情况。
- 把 block id 当 thread id 用。
- 忘记最后一个 tail block。

自检问题：

- 给定 `N=1000, block_size=256`，需要多少个 block？
- 最后一个 block 有多少合法元素？
- 如果没有 `idx < N` guard，会有什么风险？

## 4. TileLang Kernel 的几层结构

一个 TileLang kernel 通常不是直接写一个裸函数，而是分几层：

```python
@tilelang.jit
def kernel_factory(N: int, block_size: int = 256):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                C[idx] = A[idx] + B[idx]

    return kernel
```

概念对应：

- `kernel_factory`：Python 层，决定 shape、tile size、dtype 等编译期参数。
- `@tilelang.jit`：告诉 TileLang 这个 factory 要被 JIT 编译。
- `T.prim_func`：真正的 TileLang/TIR 函数。
- `T.Tensor((N,), "float32")`：声明 tensor shape 和 dtype。
- `T.Kernel`：声明 grid 和 threads。
- `T.Parallel`：声明并行工作。

为什么要 factory？

因为高性能 kernel 经常需要把 shape 和 block config 作为编译期常量。比如 GEMM 中 `block_M/block_N/block_K` 会影响 shared memory 和 Tensor Core lowering，通常不是随便运行时改的。

常见错误：

- 把 runtime tensor shape 和 compile-time shape 混在一起。
- dtype 用字符串、TileLang dtype object、PyTorch dtype 时没有转换清楚。
- 改了 config 后忘记重新 benchmark。

自检问题：

- 哪些参数应该影响编译？
- 哪些参数应该作为 tensor 输入？
- 为什么同一个 kernel 不一定适合所有 shape？

## 5. 内存层级细讲

GPU 内存不是一个平面。大致可以从慢到快理解：

| 层级 | 谁能访问 | 容量 | 速度 | 典型用途 |
| --- | --- | --- | --- | --- |
| Global memory | 所有 block | 最大 | 慢 | PyTorch tensor 输入输出 |
| Shared memory | 同一个 block | 小 | 快 | block 内 tile 复用 |
| Register/local | 单个 thread | 很小 | 最快 | 临时变量、局部累加 |
| Fragment | TileLang 抽象 | 小 | 快 | GEMM accumulator、MMA 输入输出 |

GEMM 中常见数据流：

```text
A_global tile -> A_shared
B_global tile -> B_shared
A_shared + B_shared -> C_local
C_local -> C_global
```

为什么不直接从 global 做所有计算？

因为 GEMM 中 A/B 的元素会被复用很多次。把 tile 搬到 shared memory 后，同一个 block 内多个 threads/warps 可以反复使用，减少 global memory 访问。

但是 shared memory 不是越多越好：

- shared memory 太大，会降低同一时间可驻留的 block 数。
- 访问模式不好，可能出现 bank conflict。
- tile 太大，register/fragment 压力也会上升。

常见错误：

- 以为用了 shared memory 就一定更快。
- tile 太大导致 occupancy 下降。
- shared memory shape 和实际 copy region 不匹配。
- 忘记 accumulator 初始化。

自检问题：

- 当前 kernel 的输入输出在哪里？
- 哪些数据被多个计算复用？
- 哪些数据值得搬到 shared memory？
- accumulator 是否被正确清零？

## 6. Tiling 为什么是核心

Tiling 是把大计算拆成小块。它的核心目的不是改变公式，而是改变执行方式。

以 GEMM 为例：

```text
C[M,N] = A[M,K] @ B[K,N]
```

如果一个 block 负责 `C` 的 `128x128` tile，那么：

- `block_M=128`：输出 tile 有 128 行。
- `block_N=128`：输出 tile 有 128 列。
- `block_K=32`：每次沿 K 维处理 32 个元素。

一次 K tile 的计算：

```text
A_tile: [128, 32]
B_tile: [32, 128]
C_tile: [128, 128] += A_tile @ B_tile
```

如果 `K=128`，`block_K=32`，就要循环 4 次：

```text
K tile 0: 0..31
K tile 1: 32..63
K tile 2: 64..95
K tile 3: 96..127
```

tile 参数的影响：

- tile 小：并行更多，但每个 block 计算量少，Tensor Core 利用可能不足。
- tile 大：复用更多，但 shared memory/register 压力更大。
- `block_K` 小：pipeline 粒度细，但循环次数多。
- `block_K` 大：循环次数少，但单次搬运和资源压力更高。

常见错误：

- 只换 tile size，不重新验证 correctness。
- 只看 latency，不看是否包含 JIT 编译。
- 忽略不同 shape 的 best config 可能不同。

自检问题：

- 当前输出 tile 是多大？
- 一个 block 负责多少个输出元素？
- K 维要循环几次？
- tile 改变后 shared memory 用量怎么变？

## 7. Reduction 与数值稳定

Reduction 是把多个元素合成一个结果，例如：

```text
sum(x)
max(x)
mean(x)
variance(x)
norm(x)
```

Elementwise kernel 中每个输出只依赖少数输入，互相独立。Reduction 不同，一个输出依赖一整行或一整块输入，因此要考虑：

- 多个 threads 如何共同计算一个结果。
- 中间结果放在哪里。
- 是否需要同步或分阶段归约。
- dtype 是否足够稳定。

Softmax 是 reduction 的组合：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

稳定 softmax：

```text
m = max(x)
p_i = exp(x_i - m)
out_i = p_i / sum_j p_j
```

为什么要减最大值？

因为 `exp(x)` 对大 `x` 很容易溢出。减去最大值后，最大输入变成 0，`exp(0)=1`，其它值小于等于 1。

常见错误：

- fp16 中直接累加大量元素。
- softmax 不减最大值。
- mask 后整行都是 `-inf`，没有处理特殊情况。
- tolerance 设置太紧或太松。

自检问题：

- 当前 reduction 的初始值是什么？
- 中间累加应该用什么 dtype？
- softmax 的 max 是在哪个维度上取？
- 输出和 PyTorch reference 的误差来自哪里？

## 8. Correctness Test 怎么写

开发顺序应该是：

```text
PyTorch reference -> 小 shape test -> 边界 shape test -> dtype test -> benchmark
```

一个好的 correctness test 至少包含：

- 正常 shape。
- 小 shape，方便手动检查。
- 非整除 shape。
- 多 dtype。
- 错误输入测试。

Attention 本工程默认测试方向：

```text
(1, 1, 128, 64)
(2, 8, 512, 64)
(1, 16, 1024, 128)
非整除 S，例如 129
```

比较 fp16/bf16 时，不要期待 bitwise equal。常用：

```python
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

定位错误时建议打印：

- max abs diff。
- max relative diff。
- 最大误差 index。
- 该 index 的 actual/expected。
- 相关输入片段。

常见错误：

- 只测一个 shape。
- 只测 fp32，不测 fp16/bf16。
- 忽略非整除边界。
- benchmark 之前没有先 assert correctness。

自检问题：

- 当前 test 能不能抓住 layout 错误？
- 当前 test 能不能抓住 tail block 错误？
- tolerance 是怎么选的？

## 9. Benchmark 怎么看

Benchmark 不是简单 print 一个时间。至少要记录：

- GPU 型号。
- PyTorch 版本。
- TileLang 版本。
- shape。
- dtype。
- warmup。
- repeat。
- 是否包含 JIT 编译。
- kernel config。

CUDA kernel 是异步启动的，所以不能简单这样测：

```python
start = time.time()
fn()
end = time.time()
```

本工程使用 CUDA event：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(repeat):
    fn()
end.record()
torch.cuda.synchronize()
latency_ms = start.elapsed_time(end) / repeat
```

GEMM TFLOPS：

```text
FLOPs = 2 * M * N * K
TFLOPS = FLOPs / latency_seconds / 1e12
```

Attention benchmark 要看多个 baseline：

- naive materialized attention：最基础 reference。
- PyTorch SDPA：高度优化 baseline。
- project FlashAttention API：当前项目实现。

常见错误：

- 把 JIT 编译时间算进 kernel latency。
- 没有 `torch.cuda.synchronize()`。
- 只看一个 shape 得出泛化结论。
- 没有记录 config，结果无法复现。

自检问题：

- 这次 benchmark 是否排除了首次编译？
- 结果能否被别人复现？
- 当前瓶颈更像计算还是访存？

## 10. GEMM 细讲

GEMM：

```text
C[M,N] = A[M,K] @ B[K,N]
C[i,j] = sum_k A[i,k] * B[k,j]
```

TileLang GEMM kernel 的关键结构：

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

T.clear(C_local)
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)

T.copy(C_local, C[by * block_M, bx * block_N])
```

逐行含义：

- `A_shared`：当前 block 需要的一块 A。
- `B_shared`：当前 block 需要的一块 B。
- `C_local`：当前 block 负责的输出 tile 的累加器。
- `T.clear(C_local)`：把累加器清零。
- `ko`：当前 K tile 的编号。
- `T.copy`：从 global 搬 tile 到 shared。
- `T.gemm`：用当前 A/B tile 更新 C_local。
- 最后的 `T.copy`：把 C_local 写回 global C。

为什么 `C_local` 是 `float32`？

输入通常是 `float16` 或 `bfloat16`。如果直接用低精度累加，误差会随 K 维增长变大。Tensor Core 常见路径是低精度乘法、较高精度累加。

为什么 GEMM 是 attention 基础？

Attention 的两步核心矩阵乘：

```text
scores = Q @ K^T
out = probs @ V
```

都可以看作 GEMM 或 batched GEMM。理解 GEMM 的 tiling 和 memory reuse，是理解 FlashAttention 的前置条件。

常见错误：

- A/B shape 顺序错。
- 忘记 `C_local` 清零。
- tile shape 与 `T.copy` 源区域不匹配。
- 只支持整除 shape，没有 guard。
- 误把 PyTorch 的 row-major 直觉套到所有布局。

自检问题：

- 当前 block 计算 C 的哪一块？
- A_shared 和 B_shared 分别是什么 shape？
- K loop 每次增加多少？
- C_local 为什么要用 accum dtype？

## 11. Attention 细讲

普通 attention：

```text
scores = Q @ K^T * scale
probs = softmax(scores)
out = probs @ V
```

Shape：

```text
Q:   [B, H, S, D]
K:   [B, H, S, D]
V:   [B, H, S, D]
scores: [B, H, S, S]
probs:  [B, H, S, S]
out:    [B, H, S, D]
```

显存问题来自 `S x S`。例如 `B=1, H=16, S=4096`：

```text
scores elements = 1 * 16 * 4096 * 4096 = 268,435,456
```

如果用 fp16，只 `scores` 就约 512 MB。再加 `probs`、读写带宽和反向缓存，成本会很高。

FlashAttention 的核心想法：

```text
不要完整保存 scores/probs
按 block 流式处理 K/V
维护 softmax 的 m/l/acc
最终只写 out
```

Online softmax 更新：

旧状态：

```text
m_old: 已经看过的 scores 最大值
l_old: 已经看过的 softmax 分母
acc_old: 已经看过的输出累加器
```

当前 block：

```text
scores_block = Q_block @ K_block^T * scale
m_block = max(scores_block)
m_new = max(m_old, m_block)
old_scale = exp(m_old - m_new)
p = exp(scores_block - m_new)
l_new = l_old * old_scale + sum(p)
acc_new = acc_old * old_scale + p @ V_block
```

最后：

```text
out = acc / l
```

为什么旧的 `l_old` 和 `acc_old` 要缩放？

因为新的最大值 `m_new` 可能比旧最大值更大。softmax 分母和累加器都必须用同一个基准最大值重新表达，否则不同 block 的指数不在同一个尺度上。

常见错误：

- 只更新 `l`，忘记同步缩放 `acc`。
- `m/l/acc` 用 fp16，误差过大。
- causal mask 的边界判断错。
- 最后一个 K/V block 越界。
- `D=64` 写通后，`D=128` register 压力变化没重新分析。

自检问题：

- 为什么需要 `m`、`l`、`acc` 三个状态？
- 如果新 block 的最大值更大，旧状态怎么调整？
- FlashAttention 节省的是哪一部分显存？
- causal mask 在 block 内如何判断？

## 12. FlashAttention Kernel 设计问题

开始写 TileLang FlashAttention 前，先回答这些问题：

1. 一个 block 负责哪些维度？
   - 一个 `(B,H)` 的某个 Q tile？
   - 还是把 batch/head 合并成一个 grid 维度？

2. Q block 多大？
   - `block_M` 越大，输出 tile 越大。
   - 太大可能增加 register/shared 压力。

3. K/V block 多大？
   - `block_N` 影响每次 softmax 更新的粒度。
   - 太小循环多，太大资源压力高。

4. `D` 如何处理？
   - 第一版优先 specialize `D=64`。
   - 再扩展到 `D=128`。

5. `m/l/acc` 放在哪里？
   - 通常要接近 fragment/register 级别。
   - 需要 fp32。

6. 如何处理 tail？
   - `S` 不整除 block size。
   - K/V 最后一个 block 不满。
   - Q 最后一个 block 不满。

7. 如何接入 public API？
   - `flash_attention_forward(q, k, v, causal=False, sm_scale=None)` 不变。
   - 可以内部选择 fallback 或 TileLang kernel。

8. 如何判断成功？
   - correctness 先通过。
   - non-causal 版本快于 materialized naive attention。
   - 与 PyTorch SDPA 的差距能解释。

常见错误：

- 一开始同时做 causal、D=128、多 batch、多 dtype。
- 没有 fallback reference。
- 没有小 shape debug 路线。
- benchmark 前没有跑 correctness。

自检问题：

- 第一版最小支持 shape 是什么？
- 哪些限制写进了 report？
- public API 是否保持不变？

## 13. 调试路线

遇到问题时，不要只盯着 kernel 代码。按层排查：

```text
输入检查 -> reference -> 小 shape -> dtype -> layout -> boundary -> kernel config -> benchmark
```

编译失败：

- 看 TileLang 报错位置。
- 检查 dtype 是否是 TileLang 支持的类型。
- 检查 shape 是否是编译期可知。
- 检查 `T.copy` region 是否匹配目标 buffer。

运行失败：

- 检查 tensor 是否在 CUDA。
- 检查 tensor 是否 contiguous。
- 检查输出是否正确分配。
- 检查 grid/block 是否覆盖合法范围。

结果错误：

- 先用小 shape。
- 打印最大误差 index。
- 比较中间 reference。
- 检查 boundary guard。
- 检查 accumulator 初始化。
- 检查 mask。

性能差：

- 确认不包含 JIT 编译。
- 与 PyTorch baseline 比较。
- 改一个 config 变量再测。
- 估算访存量。
- 看 tile 是否太小或太大。

自检问题：

- 当前问题属于哪一类？
- 有没有最小复现 shape？
- 有没有 PyTorch reference？
- 修复后是否补了 test？

## 14. 阶段自检总表

| Stage | 学懂后应该能回答 |
| --- | --- |
| 00 | 为什么本工程固定 TileLang 版本？为什么先跑 smoke？ |
| 00.25 | Python 的赋值、decorator、type hint、PyTorch tensor 和 C++ 直觉有什么不同？ |
| 00.5 | `QK^T -> softmax -> P@V` 每一步的 shape 和含义是什么？ |
| 01 | 一个 vector add 元素由哪个 block/thread 负责？ |
| 02 | `T.Parallel`、`T.serial`、`T.unroll` 分别适合什么场景？ |
| 03 | 编译失败、运行失败、结果错误、性能差如何区分？ |
| 04 | GEMM 中 A/B/C 如何在 global/shared/fragment 之间流动？ |
| 05 | fusion 为什么能减少 global memory traffic？ |
| 06 | online softmax 的 `m/l/acc` 分别是什么？ |
| 07 | FlashAttention 第一版为什么先做 non-causal、D=64？ |
