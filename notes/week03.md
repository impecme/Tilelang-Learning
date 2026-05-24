# Week 03 - 内存层级与调试

## Goals

- 理解 global/shared/fragment 的使用边界。
- 掌握 `T.copy` 的 tile 搬运语义。
- 能打印或查看 lowering 中间信息定位问题。

## Exercises

- 实现 correctness-first tiled matmul。
- 比较 shared memory 版本和 naive PyTorch reference。
- 故意制造一个 shape 或 dtype 错误，并记录调试过程。

## Notes

- 记录 shared memory tile shape 选择理由。
- 记录边界 shape 如何处理。

## Done Criteria

- tiled matmul 小 shape 正确。
- 有一条完整的 debug 记录：现象、定位、修复、验证。

