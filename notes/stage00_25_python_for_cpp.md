# Stage 00.25 - 面向 C++ 程序员的 Python/PyTorch 最小基础

## 阶段目标

之前主要写 C++，所以不需要从“什么是变量、什么是函数”开始，而是要补齐 Python 和 PyTorch 中最容易和 C++ 思维冲突的部分。这一阶段的目标是：

- 能读懂本工程中的 Python 文件。
- 能写简单 PyTorch reference。
- 能看懂 pytest correctness test。
- 能理解 TileLang 代码里常见的 decorator、type hint、function factory、JIT cache。
- 能避免 C++ 程序员刚学 Python 时最常见的坑。

完成这个阶段后，再进入 Transformer/Attention 背景和 TileLang DSL，会顺很多。

## 先修状态

- 熟悉 C++ 基本语法、函数、类、数组或 vector。
- 会用命令行运行程序。
Python 基础可以较薄弱，这个阶段就是为了补齐它。

## 阅读

- 本文件完整阅读。
- `scripts/check_env.py`。
- `scripts/stage00_smoke.py`。
- `tests/test_vector_add.py`。
- `kernels/vector_add.py`。
- `kernels/reference.py` 中 `_check_attention_inputs` 和 `naive_attention_forward`。
- `scripts/python_for_cpp_smoke.py`。
- `notes/glossary_zh_en.md` 第 1、2、3 节。

## 1. Python 和 C++ 的思维差异

| 主题      | C++ 常见思维           | Python 常见思维                                  |
| --------- | ---------------------- | ------------------------------------------------ |
| 编译/运行 | 先编译再运行           | 解释执行为主，也有 JIT/扩展                      |
| 类型      | 静态类型，编译期检查多 | 动态类型，运行期错误更多                         |
| 变量      | 更接近对象或内存位置   | 名字绑定到对象引用                               |
| 作用域    | 花括号 `{}`            | 缩进                                             |
| 头文件    | `.h/.hpp` 声明         | import 模块                                      |
| 内存      | RAII、指针、引用       | 引用计数 + GC，通常不手动释放                    |
| 性能      | for 循环可很快         | Python for 循环慢，重计算交给 NumPy/PyTorch/CUDA |
| 泛型      | template               | duck typing、type hint、runtime check            |
| 错误      | 返回码/异常            | 异常很常见                                       |

最重要的一点：Python 写 AI 工程时，Python 常常负责“组织计算”，真正大规模计算交给 PyTorch CUDA kernel、TileLang kernel 或 C++/CUDA 扩展。

## 2. 变量、对象、引用

Python 变量是名字，指向对象：

```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)  # [1, 2, 3, 4]
```

这不是拷贝。`a` 和 `b` 指向同一个 list。

如果想拷贝：

```python
b = a.copy()
```

PyTorch tensor 也有类似问题：

```python
y = x
```

通常不是复制数据，而是多一个名字指向同一个 tensor 对象。

真正复制：

```python
y = x.clone()
```

C++ 对照：

- Python 普通赋值更像“共享对象引用”。
- 不要把 `b = a` 理解成 C++ 中总是发生深拷贝。

## 3. 基础容器

Python 常用容器：

```python
xs = [1, 2, 3]              # list，可变
shape = (2, 8, 512, 64)     # tuple，常用于 shape
meta = {"dtype": "fp16"}    # dict
names = {"q", "k", "v"}     # set
```

在本工程中：

- shape 常用 tuple：`(B, H, S, D)`。
- 配置信息常用 dict。
- 测试参数常用 list/tuple。

索引和切片：

```python
xs = [10, 20, 30, 40]
xs[0]      # 10
xs[-1]     # 40
xs[1:3]    # [20, 30]
xs[:2]     # [10, 20]
xs[2:]     # [30, 40]
```

常见坑：

```python
rows = [[0] * 3] * 2
rows[0][0] = 1
print(rows)  # [[1, 0, 0], [1, 0, 0]]
```

两个 row 指向同一个 list。正确写法：

```python
rows = [[0] * 3 for _ in range(2)]
```

## 4. 函数、默认参数、关键字参数

Python 函数：

```python
def add(a, b):
    return a + b
```

默认参数：

```python
def flash_attention_forward(q, k, v, causal=False, sm_scale=None):
    ...
```

调用时可以用关键字参数：

```python
flash_attention_forward(q, k, v, causal=True)
```

这比纯位置参数更清楚。

常见坑：不要使用可变对象作为默认参数：

```python
def bad(xs=[]):
    xs.append(1)
    return xs
```

默认参数只创建一次。更安全：

```python
def good(xs=None):
    if xs is None:
        xs = []
    xs.append(1)
    return xs
```

本工程中 `sm_scale=None` 就是这种常见模式：`None` 表示“没有显式传入，函数内部决定默认值”。

## 5. Type Hint 是提示，不是 C++ 类型系统

Python 可以写类型标注：

```python
def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b
```

这有三个作用：

- 给人读。
- 给 IDE 和静态检查工具看。
- 让 API 更清楚。

但它不是 C++ 那种强制编译期类型检查。运行时仍然可能传错类型，所以本工程很多函数还会显式检查：

```python
if a.ndim != 2:
    raise ValueError("a must be rank-2")
```

看到：

```python
float | None
```

意思是这个值可以是 `float`，也可以是 `None`。

## 6. Module、Import 和入口函数

Python 文件就是 module。

```python
from kernels.reference import naive_attention_forward
```

意思是从 `kernels/reference.py` 里导入函数。

导入时，Python 会执行这个文件中的顶层代码，并把里面定义的函数、类、变量放到模块对象里。为了避免“只是 import 一个函数，却顺便跑了整个脚本”，Python 程序常用入口保护。

常见入口写法：

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

这里有三个概念：

- `main()`：普通函数名，不是 Python 语法关键字，只是约定俗成地表示“脚本主流程”。
- `__name__`：当前文件作为模块时的名字。
- `"__main__"`：当前文件被直接运行时，Python 给 `__name__` 设置的特殊值。

直接运行一个文件：

```bash
python3 scripts/python_for_cpp_smoke.py
```

此时该文件内部：

```python
__name__ == "__main__"
```

所以会执行：

```python
main()
```

如果这个文件被别的代码 import：

```python
import scripts.python_for_cpp_smoke
```

此时它的 `__name__` 通常是：

```text
scripts.python_for_cpp_smoke
```

不等于 `"__main__"`，因此不会自动执行 `main()`。

这不是在判断“文件名是不是 main”，而是在判断“当前模块是不是正在作为程序入口运行”。

C++ 对照：

- C++ 的 `int main()` 是语言规定的程序入口。
- Python 的 `main()` 只是普通函数。
- `if __name__ == "__main__": main()` 是一种手动指定脚本入口的习惯写法。

这个写法的好处：

- 文件可以直接当脚本运行。
- 文件也可以被其它模块 import 复用函数。
- import 时不会误触发 benchmark、测试、打印或耗时逻辑。

本工程里的脚本和 benchmark 都会用这种模式。

## 7. Decorator：`@something` 是什么

Python decorator 是“把函数交给另一个对象处理”的语法。通用规则是：

```python
@decorator
def f():
    ...
```

大致等价于：

```python
def f():
    ...

f = decorator(f)
```

也就是说，`decorator` 会接收原来的函数 `f`，然后返回一个新的函数或对象。至于它到底是在“包装函数”“打标记”“注册函数”，还是“交给 JIT 编译器接管”，取决于这个 decorator 自己怎么实现。

比如：

```python
@pytest.mark.cuda
def test_x():
    ...
```

这里更像是把 `test_x` 交给 `pytest.mark.cuda` 标记一下，让 pytest 知道这是一个 CUDA 相关测试。

TileLang 中：

```python
@tilelang.jit
def kernel_factory(...):
    ...
```

含义：这个函数不是普通 Python 函数那么简单，它会被 TileLang 接管，用于 JIT 编译 kernel。

还有：

```python
@T.prim_func
def kernel(...):
    ...
```

含义：这个函数体会按 TileLang/TIR 的规则解释，不是普通 Python 执行逻辑。

C++ 对照：

- decorator 有点像“给函数加编译/运行时属性或包装器”。
- 但它比 C++ attribute 更强，因为它可以返回一个新函数或对象。

## 8. Function Factory 和 Closure

TileLang 里常见这种结构：

```python
def _compile_vector_add(numel: int, block: int, dtype: str):
    @tilelang.jit
    def _kernel_factory(N: int, block_size: int = 256):
        @T.prim_func
        def _kernel(...):
            ...
        return _kernel

    return _kernel_factory(numel, block)
```

这叫 function factory：函数创建函数。

为什么这么写？

因为 TileLang kernel 经常依赖编译期参数，例如：

- `N`
- `block_size`
- dtype
- tile size
- threads
- num_stages

这些参数进入 factory 后，TileLang 可以生成对应配置的 kernel。

Closure 是内部函数捕获外部变量。暂时不需要一开始理解所有细节，但要知道：Python 函数可以作为值被创建、传递、缓存和返回。

## 9. `@lru_cache`：为什么编译函数要缓存

本工程里有：

```python
@lru_cache(maxsize=32)
def _compile_vector_add(...):
    ...
```

意思是：同一组参数编译过一次后，下次直接复用结果。

这对 TileLang 很重要，因为 JIT 编译有成本。第一次运行可能慢，后面运行才更接近真实 kernel latency。

常见误解：

- 第一次运行慢不代表 kernel 慢，可能是在编译。
- 改了 shape/config 后，可能重新触发编译。

## 10. 异常处理

Python 常见错误处理：

```python
if a.shape != b.shape:
    raise ValueError("a and b must have the same shape")
```

测试里可以检查异常：

```python
with pytest.raises(ValueError):
    vector_add_reference(a, b)
```

本工程会用异常表达输入不合法，例如：

- shape 不匹配。
- dtype 不支持。
- tensor 不在 CUDA。
- tensor 不 contiguous。

## 11. PyTorch Tensor 最小基础

PyTorch 的核心对象是 `torch.Tensor`。可以先把 tensor 理解成“带元信息的多维数组”。

和 C++ 里的 `std::vector<float>` 相比，tensor 不只是保存一段数据，它还保存：

- `shape`：每个维度有多长。
- `dtype`：每个元素是什么类型。
- `device`：数据在 CPU 还是 GPU。
- `stride`：每个维度移动 1 个 index 时，底层存储跳多少个元素。
- `requires_grad`：是否需要 autograd 记录梯度。本工程主要写 inference/reference，通常不需要梯度。

最小创建方式：

```python
import torch

x = torch.randn((2, 3))
y = torch.zeros((2, 3))
z = torch.ones((2, 3))
```

常见创建函数：

| 写法                  | 含义               | 常见用途                 |
| --------------------- | ------------------ | ------------------------ |
| `torch.randn(shape)`  | 正态分布随机数     | 构造测试输入             |
| `torch.zeros(shape)`  | 全 0               | 初始化输出或 mask        |
| `torch.ones(shape)`   | 全 1               | 构造简单参考输入         |
| `torch.empty(shape)`  | 未初始化内存       | 性能路径中先分配，后写入 |
| `torch.tensor([...])` | 从 Python 数据创建 | 小例子、手算验证         |

`torch.empty` 不会把内容清零：

```python
x = torch.empty((2, 3))
print(x)
```

它里面可能是任意旧值。只有确定后面会完整写入时，才适合用 `empty`。

常用属性可以直接打印：

```python
import torch

x = torch.randn((2, 3), dtype=torch.float32)

x.shape
x.dtype
x.device
x.ndim
x.is_cuda
x.is_contiguous()
x.requires_grad
```

这些属性的含义：

- `x.shape`：形状，例如 `torch.Size([2, 3])`，很像 tuple。
- `x.ndim`：维度数量，也叫 rank；`(2, 3)` 是 rank-2。
- `x.dtype`：数据类型，例如 `torch.float32`、`torch.float16`、`torch.bfloat16`。
- `x.device`：所在设备，例如 `cpu` 或 `cuda:0`。
- `x.is_cuda`：是否在 CUDA GPU 上。
- `x.is_contiguous()`：底层内存是否按默认连续布局排列。
- `x.requires_grad`：是否参与 autograd。算子 correctness reference 通常不需要它。

CPU tensor 和 CUDA tensor 的区别：

```python
cpu_x = torch.randn((2, 3), device="cpu")

if torch.cuda.is_available():
    cuda_x = torch.randn((2, 3), device="cuda", dtype=torch.float16)
    print(cuda_x.device)
```

CPU tensor 的数据在主机内存里，CUDA tensor 的数据在 GPU 显存里。TileLang kernel 需要 CUDA tensor；如果传 CPU tensor，通常应该在 Python 包装层直接报错。

改变 dtype/device 的常见写法：

```python
x = torch.randn((2, 3))

x_fp32 = x.float()
x_half = x.half()
x_bf16 = x.to(dtype=torch.bfloat16)

if torch.cuda.is_available():
    x_cuda = x.to("cuda")
    x_back = x_cuda.cpu()
```

注意：`.to(...)`、`.float()`、`.half()`、`.cpu()` 通常返回新 tensor，不一定原地修改。

```python
x = torch.randn((2, 3))
y = x.float()

print(x is y)  # 不要依赖它一定 True 或 False
```

学习时更稳的写法是把结果接住：

```python
x = x.to(dtype=torch.float32)
```

本工程写 PyTorch reference 时，常见习惯是先转成 fp32 做稳定计算：

```python
scores = q.float() @ k.float().transpose(-2, -1)
```

这样 fp16/bf16 输入也能有更稳定的参考结果。

## 12. PyTorch Shape 操作

shape 是 PyTorch 代码里最重要的调试线索。只要 shape 推导清楚，很多 attention/GEMM 问题会立刻变简单。

索引和切片：

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)

print(x.shape)       # torch.Size([2, 3, 4])
print(x[0].shape)    # torch.Size([3, 4])
print(x[:, 1:].shape)  # torch.Size([2, 2, 4])
print(x[..., -1].shape)  # torch.Size([2, 3])
```

这里：

- `x[0]`：取第 0 维的第一个元素，维度数量减少 1。
- `x[:, 1:]`：第 0 维全取，第 1 维从 1 取到结尾。
- `...`：省略中间维度。
- `-1`：最后一个 index，或最后一个维度。

`dim=-1` 表示最后一维：

```python
x = torch.randn((2, 3, 4))

print(x.sum(dim=-1).shape)  # (2, 3)
print(x.max(dim=-1).values.shape)  # (2, 3)
```

在 softmax 中：

```python
import torch

scores = torch.randn((2, 8, 4, 4))
p = torch.softmax(scores, dim=-1)
print(p.shape)
print(p.sum(dim=-1))
```

意思是对最后一维做 softmax。Attention 的 `scores.shape = (B, H, S, S)`，最后一维表示“当前 query 对所有 key 的分数”，所以 softmax 要沿最后一维做。

常见 shape 操作：

```python
import torch

x = torch.arange(24)
x3 = x.reshape(2, 3, 4)
xt = x3.transpose(-2, -1)
xp = x3.permute(0, 2, 1)
xc = xt.contiguous()
xv = x3.view(2, 12)

print(x3.shape, xt.shape, xp.shape, xc.shape, xv.shape)
```

区别：

- `reshape`：改逻辑形状；尽量返回 view，不行就复制。
- `view`：也改逻辑形状，但要求内存布局兼容；非 contiguous 时经常失败。
- `transpose`：交换两个维度。
- `permute`：按指定顺序重排多个维度。
- `contiguous`：重新整理底层内存，使布局变成默认连续。

stride 是理解 contiguous 的关键：

```python
import torch

x = torch.arange(24).reshape(2, 3, 4)
xt = x.transpose(-2, -1)

print(x.shape, x.stride(), x.is_contiguous())
print(xt.shape, xt.stride(), xt.is_contiguous())

xc = xt.contiguous()
print(xc.shape, xc.stride(), xc.is_contiguous())
```

`transpose` 通常只是改变“如何解释底层存储”，不一定真的搬数据。因此逻辑 shape 变了，但底层数据不一定重新排列。TileLang kernel 常常假设 contiguous 输入，所以包装层需要检查：

```python
if not x.is_contiguous():
    raise ValueError("expected contiguous tensor")
```

Attention 里常见：

```python
k.transpose(-2, -1)
```

如果：

```python
k.shape = (B, H, S, D)
```

那么：

```python
k.transpose(-2, -1).shape = (B, H, D, S)
```

这样才能做：

```python
q @ k.transpose(-2, -1)
```

逐步看：

```text
q.shape = (B, H, S, D)
k.shape = (B, H, S, D)
k.transpose(-2, -1).shape = (B, H, D, S)
q @ k.transpose(-2, -1) = (B, H, S, D) @ (B, H, D, S)
scores.shape = (B, H, S, S)
```

## 13. PyTorch 矩阵乘法

PyTorch 里 `@` 是矩阵乘法运算符，常用来写 `torch.matmul`：

```python
C = A @ B
```

可以理解为：

```python
C = torch.matmul(A, B)
```

二维矩阵乘法规则：

```python
import torch

A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B
print(C.shape)  # (2, 4)
```

shape 规则：

```text
A.shape = (M, K)
B.shape = (K, N)
C.shape = (M, N)
```

中间的 `K` 必须相同。每个输出元素是：

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

Batched matmul 是把前面的维度当成 batch：

```python
Q = torch.randn(2, 8, 512, 64)
K = torch.randn(2, 8, 512, 64)
scores = Q @ K.transpose(-2, -1)
print(scores.shape)  # (2, 8, 512, 512)
```

PyTorch 会把前面的维度 `(2, 8)` 当成 batch 维，对每个 batch 独立做矩阵乘。

这就是 Attention 中 `QK^T` 的 PyTorch 写法。

更详细地拆：

```text
Q[b, h].shape = (512, 64)
K[b, h].shape = (512, 64)
K[b, h].T.shape = (64, 512)
Q[b, h] @ K[b, h].T = (512, 64) @ (64, 512)
scores[b, h].shape = (512, 512)
```

因此整体：

```text
scores.shape = (B, H, S, S)
```

常见错误：

- 忘记 `transpose(-2, -1)`，导致 `(S, D) @ (S, D)` 维度不匹配。
- 把 `(B, S, H, D)` 当成 `(B, H, S, D)`。
- 只看最后结果 shape，没有确认每个 batch/head 是否对应正确。

## 14. Broadcasting

Broadcasting 是 PyTorch 自动扩展维度的规则。

例子：

```python
x = torch.randn(2, 3)
bias = torch.randn(3)
y = x + bias
```

`bias` 会自动按第 0 维扩展，相当于每一行都加同一个 bias。

规则可以这样记：

```text
从最后一维开始对齐；
两个维度相等，可以广播；
其中一个维度是 1，可以扩展到另一个维度；
缺失的前置维度可以当作 1。
```

例子：

```python
import torch

x = torch.randn(2, 3)
a = torch.randn(3)
b = torch.randn(2, 1)

print((x + a).shape)  # (2, 3)
print((x + b).shape)  # (2, 3)
```

`a.shape = (3,)` 会被看成 `(1, 3)`，扩展到 `(2, 3)`。

`b.shape = (2, 1)` 的最后一维是 1，可以扩展到 3。

Attention 中 scale：

```python
import math
import torch

scores = torch.randn((2, 8, 4, 4))
scale = 1.0 / math.sqrt(64)
scores = scores * scale
print(scores.shape)
```

`scale` 是一个 Python float，会广播到整个 tensor。

mask 也经常依赖 broadcasting：

```python
scores = torch.randn(2, 8, 4, 4)
mask = torch.ones(4, 4, dtype=torch.bool)

masked_scores = scores.masked_fill(~mask, float("-inf"))
print(masked_scores.shape)  # (2, 8, 4, 4)
```

这里 `mask.shape = (4, 4)` 会广播到 `(2, 8, 4, 4)`。这可能正是想要的，也可能是 bug：如果每个 batch/head 应该有不同 mask，就不能只传 `(4, 4)`。

常见坑：

- shape 不小心 broadcast 成合法但语义错误的结果。
- 以为发生了复制，实际只是按规则虚拟扩展。
- 只看没有报错，就以为语义正确。

## 15. PyTorch 和 Python for 循环

Python for 循环慢，PyTorch tensor 操作快，因为大计算在底层 C++/CUDA 中执行。

不推荐：

```python
for i in range(N):
    y[i] = x[i] + 1
```

推荐：

```python
y = x + 1
```

`y = x + 1` 看起来是一行 Python，但真正的大量元素计算发生在 PyTorch 的底层 C++/CUDA kernel 里。Python 只负责发起这个操作。

这也是 AI 工程里常说的调用链：

```text
Python API -> PyTorch dispatcher -> CUDA/C++ backend -> GPU kernel -> GPU hardware
```

什么时候可以用 Python loop？

- debug 小 shape。
- 手写很朴素的 reference。
- 打印中间值，帮助理解公式。

什么时候不应该用 Python loop？

- 大 tensor 的性能路径。
- benchmark 中被测函数的核心计算。
- 本来可以用 PyTorch/Tensor/TileLang/CUDA 表达的逐元素或矩阵计算。

一个 correctness-first reference 可以先这样写：

```python
import torch


def row_sum_reference_slow(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")

    out = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i] += x[i, j].float()
    return out
```

但更推荐的 PyTorch reference 是：

```python
import torch


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return x.float().sum(dim=-1)
```

第一版适合理解，第二版更接近工程写法。

## 16. pytest 最小基础

测试函数通常这样写：

```python
def test_vector_add_reference():
    a = torch.randn(32)
    b = torch.randn(32)
    torch.testing.assert_close(vector_add_reference(a, b), a + b)
```

pytest 会自动发现以 `test_` 开头的函数。

跳过测试：

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_kernel():
    ...
```

参数化测试：

```python
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_x(dtype):
    ...
```

比较浮点结果：

```python
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

`assert_close` 不是要求完全相等，而是检查：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
```

其中：

- `atol`：绝对误差容忍。
- `rtol`：相对误差容忍。

fp32 通常可以设得更严格；fp16/bf16 因为精度低，通常需要更宽松：

```python
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

本工程写 reference test 时，建议固定模板：

```python
import torch


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return x.float().sum(dim=-1)


def test_row_sum_reference():
    x = torch.randn((4, 8), dtype=torch.float16)

    actual = row_sum_reference(x)
    expected = x.float().sum(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
```

如果要测试 CUDA：

```python
import pytest
import torch


def row_sum_reference(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be rank-2")
    return x.float().sum(dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_row_sum_reference_cuda():
    x = torch.randn((4, 8), device="cuda", dtype=torch.float16)
    actual = row_sum_reference(x)
    expected = x.float().sum(dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
```

算子开发里，PyTorch reference 的作用是给 TileLang/CUDA kernel 提供“标准答案”。开发顺序建议固定：

```text
写 PyTorch reference -> 写 correctness test -> 写 TileLang kernel -> 对比 actual/expected
```

## 17. 本工程中最需要先读懂的 Python 语法

优先读懂这些：

```python
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import torch
```

```python
def f(x: torch.Tensor) -> torch.Tensor:
    ...
```

```python
if x.ndim != 4:
    raise ValueError(...)
```

```python
@pytest.mark.parametrize(...)
@pytest.mark.skipif(...)
```

```python
@tilelang.jit
@T.prim_func
```

```python
if __name__ == "__main__":
    main()
```

读懂这些，就能开始读本项目的大部分代码。

## 18. C++ 程序员常见 Python 坑

1. 赋值不是深拷贝：

   ```python
   b = a
   ```

2. 默认可变参数会被复用：

   ```python
   def f(xs=[]): ...
   ```

3. `/` 和 `//` 不同：

   ```python
   5 / 2   # 2.5
   5 // 2  # 2
   ```

4. `range(n)` 不包含 `n`：

   ```python
   range(3)  # 0,1,2
   ```

5. `and/or/not` 不是 `&&/||/!`。

6. Python 没有花括号作用域，靠缩进。

7. Tensor 的 transpose 可能产生非 contiguous tensor。

8. CUDA 操作通常异步，benchmark 需要 synchronize 或 CUDA event。

9. Python loop 慢，不要用它写性能核心。

10. type hint 不是强制编译期类型检查。

## 19. 小练习

1. Python 基础：
   - 写一个函数 `ceildiv(a, b)`。
   - 输入 `a=1000, b=256`，输出 `4`。

2. Tensor 基础：
   - 创建 `x = torch.randn((2, 3, 4))`。
   - 打印 `shape/dtype/device/ndim/stride/is_contiguous/requires_grad`。
   - 分别创建 `zeros/ones/empty/tensor`，观察初始值差异。
   - 如果有 CUDA，做 `x_cuda = x.to("cuda")`，打印 `x.device` 和 `x_cuda.device`。
   - 做 `x_half = x.to(dtype=torch.float16)`，确认原始 `x.dtype` 是否改变。
   - 做 `x.transpose(-2, -1)`，打印 shape 和 `is_contiguous()`。
   - 对 transpose 结果调用 `.contiguous()`，再次打印 `stride()` 和 `is_contiguous()`。

3. Matmul 基础：
   - 创建 `A=(2,3)`，`B=(3,4)`。
   - 计算 `C=A@B`。
   - 手动写出 `C.shape` 为什么是 `(2,4)`。

4. Attention shape：
   - 创建 `Q,K,V = torch.randn((2,8,16,64))`。
   - 计算 `scores = Q @ K.transpose(-2, -1)`。
   - 打印 `scores.shape`。
   - 解释为什么是 `(2,8,16,16)`。

5. Broadcasting：
   - 创建 `x = torch.randn((2, 3))`。
   - 分别创建 `a = torch.randn(3)` 和 `b = torch.randn(2, 1)`。
   - 计算 `x + a`、`x + b`，写出广播前后的 shape。
   - 尝试 `x + torch.randn(2)`，观察报错并解释最后一维为什么对不上。

6. 小型 PyTorch reference：
   - 实现 `row_sum_reference(x)`。
   - 要求 `x.ndim == 2`，否则 `raise ValueError`。
   - 返回 `x.float().sum(dim=-1)`。
   - 用 `torch.testing.assert_close` 和 PyTorch 原生表达式对比。

7. pytest：
   - 给 `ceildiv` 写一个 test。
   - 故意写错一次，观察 pytest 输出。
   - 给 `row_sum_reference` 增加一个正常输入 test 和一个错误 rank test。

8. 入口保护：
   - 找到 `scripts/python_for_cpp_smoke.py` 末尾的 `if __name__ == "__main__":`。
   - 解释直接运行脚本时为什么会进入 `main()`。
   - 解释被其它文件 import 时为什么不应该自动跑完整 demo。

9. 运行这个阶段的配套脚本：

   ```bash
   python3 scripts/python_for_cpp_smoke.py
   ```

   观察：
   - Python list 赋值后的共享引用。
   - decorator 如何包装函数。
   - tensor 的 `shape/dtype/device/ndim`。
   - `transpose(-2, -1)` 为什么会改变 contiguous 状态。
   - `Q @ K.transpose(-2, -1)` 和 `P @ V` 的 shape。

## 20. 阶段验收

完成这个阶段后，应能回答：

- Python 的 `b = a` 和 C++ 拷贝有什么不同？
- `@pytest.mark.parametrize` 是什么？
- `@tilelang.jit` 为什么不是普通函数调用？
- `__name__ == "__main__"` 为什么能区分直接运行和 import？
- `torch.Tensor.shape/dtype/device/stride/contiguous` 分别是什么？
- `rank`、`ndim`、`dim=-1` 分别是什么意思？
- `.to("cuda")`、`.cpu()`、`.float()`、`.half()` 为什么要接住返回值？
- `transpose(-2, -1)` 对 `(B,H,S,D)` 做了什么？
- `reshape`、`view`、`transpose`、`permute`、`contiguous` 的区别是什么？
- broadcasting 为什么可能让 shape 合法但语义错误？
- 能否写一个简单 PyTorch reference test，例如 `row_sum_reference`？
- 为什么 PyTorch for 循环不是写高性能算子的方式？
- 为什么 CUDA benchmark 不能只用普通 `time.time()`？
