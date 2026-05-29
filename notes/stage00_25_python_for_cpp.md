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

| 主题 | C++ 常见思维 | Python 常见思维 |
| --- | --- | --- |
| 编译/运行 | 先编译再运行 | 解释执行为主，也有 JIT/扩展 |
| 类型 | 静态类型，编译期检查多 | 动态类型，运行期错误更多 |
| 变量 | 更接近对象或内存位置 | 名字绑定到对象引用 |
| 作用域 | 花括号 `{}` | 缩进 |
| 头文件 | `.h/.hpp` 声明 | import 模块 |
| 内存 | RAII、指针、引用 | 引用计数 + GC，通常不手动释放 |
| 性能 | for 循环可很快 | Python for 循环慢，重计算交给 NumPy/PyTorch/CUDA |
| 泛型 | template | duck typing、type hint、runtime check |
| 错误 | 返回码/异常 | 异常很常见 |

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

常见入口写法：

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

含义：

- 直接运行这个文件时，执行 `main()`。
- 被其它文件 import 时，不自动执行 `main()`。

本工程里的脚本和 benchmark 都会用这种模式。

## 7. Decorator：`@something` 是什么

Python decorator 是“包装函数”的语法。比如：

```python
@pytest.mark.cuda
def test_x():
    ...
```

等价于把 `test_x` 交给 `pytest.mark.cuda` 标记一下。

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

创建 tensor：

```python
import torch

x = torch.randn((2, 3))
y = torch.zeros((2, 3))
```

CUDA tensor：

```python
x = torch.randn((2, 3), device="cuda", dtype=torch.float16)
```

常用属性：

```python
x.shape
x.dtype
x.device
x.ndim
x.is_cuda
x.is_contiguous()
```

改变 dtype/device：

```python
x_fp32 = x.float()
x_cuda = x.to("cuda")
x_half = x.to(dtype=torch.float16)
```

注意：`.to(...)` 通常返回新 tensor，不一定原地修改。

## 12. PyTorch Shape 操作

常见操作：

```python
x.reshape(2, 3, 4)
x.transpose(-2, -1)
x.permute(0, 2, 1)
x.contiguous()
x.view(2, 12)
```

区别：

- `reshape`：尽量返回 view，不行就复制。
- `view`：要求内存布局兼容。
- `transpose/permute`：改变维度顺序，常导致非 contiguous。
- `contiguous`：重新整理内存，使布局连续。

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

## 13. PyTorch 矩阵乘法

二维 matmul：

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B
print(C.shape)  # (2, 4)
```

Batched matmul：

```python
Q = torch.randn(2, 8, 512, 64)
K = torch.randn(2, 8, 512, 64)
scores = Q @ K.transpose(-2, -1)
print(scores.shape)  # (2, 8, 512, 512)
```

PyTorch 会把前面的维度 `(2, 8)` 当成 batch 维，对每个 batch 独立做矩阵乘。

这就是 Attention 中 `QK^T` 的 PyTorch 写法。

## 14. Broadcasting

Broadcasting 是 PyTorch 自动扩展维度的规则。

例子：

```python
x = torch.randn(2, 3)
bias = torch.randn(3)
y = x + bias
```

`bias` 会自动按第 0 维扩展，相当于每一行都加同一个 bias。

Attention 中 scale：

```python
scores = scores * scale
```

`scale` 是一个 Python float，会广播到整个 tensor。

常见坑：

- shape 不小心 broadcast 成合法但语义错误的结果。
- 以为发生了复制，实际只是按规则虚拟扩展。

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

但写 reference 或 debug 小 shape 时，Python loop 可以接受。写性能路径时，应交给 PyTorch/TileLang/CUDA。

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
   - 打印 `shape/dtype/device/ndim`。
   - 做 `x.transpose(-2, -1)`，打印 shape 和 `is_contiguous()`。

3. Matmul 基础：
   - 创建 `A=(2,3)`，`B=(3,4)`。
   - 计算 `C=A@B`。
   - 手动写出 `C.shape` 为什么是 `(2,4)`。

4. Attention shape：
   - 创建 `Q,K,V = torch.randn((2,8,16,64))`。
   - 计算 `scores = Q @ K.transpose(-2, -1)`。
   - 打印 `scores.shape`。
   - 解释为什么是 `(2,8,16,16)`。

5. pytest：
   - 给 `ceildiv` 写一个 test。
   - 故意写错一次，观察 pytest 输出。

6. 运行这个阶段的配套脚本：

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
- `torch.Tensor.shape/dtype/device` 分别是什么？
- `transpose(-2, -1)` 对 `(B,H,S,D)` 做了什么？
- 为什么 PyTorch for 循环不是写高性能算子的方式？
- 为什么 CUDA benchmark 不能只用普通 `time.time()`？
