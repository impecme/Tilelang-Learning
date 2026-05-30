"""Stage metadata for the TileLang learning lab.

This module is the single source of truth for the learning stages.  The
health-check script, tests, and documents all refer to these ids and goals so
the course structure stays consistent as the project grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class StageSpec:
    id: str
    order: int
    name: str
    path: str
    level: str
    goal: str
    prerequisites: tuple[str, ...]
    core_concepts: tuple[str, ...]
    kernels: tuple[str, ...]
    run_commands: tuple[str, ...]
    acceptance_checks: tuple[str, ...]
    deliverables: tuple[str, ...]
    common_mistakes: tuple[str, ...]
    next_gate: str


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        id="basic",
        order=1,
        name="01 Basic Kernels",
        path="01_basic_kernels",
        level="基础阶段",
        goal=(
            "读懂并修改最小 TileLang kernel，掌握 T.Kernel、T.Parallel、"
            "全局下标、boundary guard、dtype/device/contiguous 检查。"
        ),
        prerequisites=(
            "会运行 pytest 和 python3 -m 模块命令",
            "知道 PyTorch Tensor 的 shape、dtype、device",
            "理解一维数组和二维矩阵的基本索引",
        ),
        core_concepts=(
            "@tilelang.jit",
            "T.Kernel",
            "T.Parallel",
            "T.ceildiv",
            "global index",
            "boundary guard",
            "elementwise",
            "serial reduction",
            "single-block parallel reduction",
        ),
        kernels=("copy", "vector_add", "axpy", "row_sum", "row_sum_parallel"),
        run_commands=(
            "python3 scripts/check_project.py --stage basic",
            "python3 scripts/run_lab.py --lab basic",
            "pytest tests/test_basic.py -q",
            "RUN_TILELANG_SMOKE=1 pytest tests/test_basic.py -q -m tilelang",
            "python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2",
            "python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv",
        ),
        acceptance_checks=(
            "能解释 idx = bx * block_size + i 的来源",
            "能说明 N=1000 时为什么最后一个 block 需要 boundary guard",
            "能区分 elementwise 和 reduction 的数据依赖",
            "能解释 row_sum_parallel_tilelang 的 cols <= block_n 限制",
            "能把 block_size 改成另一个值并保持 N=1024/N=1000 都通过",
        ),
        deliverables=(
            "完成 reports/stage01_basic_template.md 的一份学习记录",
            "记录至少一次整除 shape 和一次非整除 shape 的测试结果",
            "用自己的话写出 copy/vector_add/axpy/row_sum 的输入输出语义",
            "记录一次 row_sum 串行版与 parallel 版 benchmark 对比",
        ),
        common_mistakes=(
            "忘记 .contiguous() 导致 kernel 收到非连续 tensor",
            "把 block 内局部下标 i 当成全局下标",
            "只测试 N=1024，漏掉 N=1000 的尾块场景",
            "把 row_sum 当成每个元素独立计算的 elementwise kernel",
        ),
        next_gate=(
            "只有当你能独立解释边界保护，并能根据测试报错定位 shape/dtype/device 问题时，"
            "再进入 advanced 阶段。"
        ),
    ),
    StageSpec(
        id="advanced",
        order=2,
        name="02 Advanced Ops",
        path="02_advanced_ops",
        level="进阶阶段",
        goal=(
            "理解 GEMM、softmax、RMSNorm、GELU 和 fusion 的 TileLang 数据流，"
            "能解释 shared memory、fragment、reduction、数值稳定和 benchmark 结果。"
        ),
        prerequisites=(
            "已完成 basic 阶段的测试和学习记录",
            "能读懂二维矩阵乘法 C = A @ B",
            "知道 fp16/fp32 误差容忍和 torch.testing.assert_close",
        ),
        core_concepts=(
            "shared memory",
            "fragment accumulator",
            "T.copy",
            "T.gemm",
            "T.clear",
            "tile-aligned shape",
            "stable softmax",
            "fp32 accumulation",
            "operator fusion",
            "single-block reduction",
            "T.reduce_sum",
            "T.reduce_max",
        ),
        kernels=(
            "gemm",
            "row_softmax",
            "row_softmax_parallel",
            "rmsnorm",
            "rmsnorm_parallel",
            "gelu",
            "linear_bias_gelu",
        ),
        run_commands=(
            "python3 scripts/check_project.py --stage advanced",
            "python3 scripts/run_lab.py --lab gemm",
            "python3 scripts/run_lab.py --lab reduction",
            "pytest tests/test_advanced.py -q",
            "RUN_TILELANG_SMOKE=1 pytest tests/test_advanced.py -q -m tilelang",
            "python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 --warmup 1 --repeat 2",
            "python3 -m benchmarks.bench_optimization --run-tilelang --warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv",
            "python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 --warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv",
        ),
        acceptance_checks=(
            "能说清 M/N/K 分别来自 A、B、C 的哪一维",
            "能解释 global tile -> shared memory -> fragment -> global 的 GEMM 数据流",
            "能说明 softmax 为什么要减去每行最大值",
            "能解释 v1 GEMM 为什么拒绝非 tile 对齐 shape",
            "能用 tolerance 解释 fp16 GEMM 和 fp32 reference 的误差",
            "能读懂 benchmark CSV，并解释 scale/add_bias/scale_causal_mask 的性能差异",
            "能解释串行 reduction 和 parallel reduction 的 latency 差异",
        ),
        deliverables=(
            "完成 reports/stage02_advanced_template.md 的一份学习记录",
            "记录 gemm/softmax/rmsnorm/gelu 至少各一次 correctness 结果",
            "记录一次 benchmark，并判断瓶颈更偏 memory-bound 还是 compute-bound",
            "记录一次 optimization CSV，并写下下一轮只改变一个变量的实验计划",
            "完成 reports/reduction_optimization_template.md 的一份 reduction 对比记录",
        ),
        common_mistakes=(
            "把 GEMM 的 K 维循环理解成输出维度",
            "忘记 T.clear 导致 fragment 累加器保留旧值",
            "直接 exp(x) 造成 softmax overflow",
            "用非 tile 对齐 shape 调 gemm_tilelang 却期待自动处理 tail tile",
            "把教学版串行 reduction 当成高性能最终写法",
        ),
        next_gate=(
            "只有当你能画出 GEMM 数据流，并能解释 softmax/RMSNorm 的 reduction 过程时，"
            "再进入 decoder 阶段。"
        ),
    ),
    StageSpec(
        id="decoder",
        order=3,
        name="03 Decoder Block Pipeline",
        path="03_decoder_block_pipeline",
        level="大型综合阶段",
        goal=(
            "把前两阶段算子组合成单 Decoder Block + LM Head logits，"
            "能沿着 shape 跟踪 RMSNorm、QKV、causal attention、MLP、residual 和 logits。"
        ),
        prerequisites=(
            "已完成 advanced 阶段的测试和学习记录",
            "能解释 GEMM、softmax、RMSNorm 的输入输出 shape",
            "知道 self-attention 中 Q/K/V 的含义",
        ),
        core_concepts=(
            "MiniDecoderConfig",
            "MiniDecoderWeights",
            "QKV projection",
            "head split",
            "causal mask",
            "attention probabilities",
            "residual add",
            "MLP",
            "LM head logits",
            "pipeline validation",
            "optimized reduction path",
        ),
        kernels=(
            "rmsnorm",
            "gemm",
            "causal_mask",
            "row_softmax",
            "decoder_block_optimized",
            "vector_add",
            "linear_bias_gelu",
            "decoder_block",
        ),
        run_commands=(
            "python3 scripts/check_project.py --stage decoder",
            "python3 scripts/run_lab.py --lab decoder",
            "pytest tests/test_decoder.py -q",
            "RUN_TILELANG_SMOKE=1 pytest tests/test_decoder.py -q -m tilelang",
            "python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --warmup 1 --repeat 1",
            "python3 -m benchmarks.bench_decoder_block --seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 --run-tilelang --compare-optimized --warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv",
        ),
        acceptance_checks=(
            "能从 x=(B,S,H) 推导 q/k/v=(B,num_heads,S,head_dim)",
            "能解释 QK^T、softmax、P@V 的 shape 变化",
            "能说明 causal mask 屏蔽未来 token 的原因",
            "能解释两次 RMSNorm 和两次 residual add 的位置",
            "能让 mini_inference_tilelang 的 logits 与 reference close",
            "能让 mini_inference_tilelang_optimized 的 logits 与 reference close",
            "能指出 Decoder TileLang 路径中哪些小操作已经去 PyTorch 化",
            "能说明 optimized 入口只替换 RMSNorm 和 softmax reduction",
        ),
        deliverables=(
            "完成 reports/stage03_decoder_template.md 的一份学习记录",
            "画出 Decoder Block 的 shape 流程图或文字版流程",
            "记录 decoder block 输出和 logits 的 correctness 结果",
            "记录一次 decoder benchmark CSV，并解释优化前后的编排差异",
            "记录一次 decoder optimized reductions 的 before/after latency",
        ),
        common_mistakes=(
            "混淆 hidden_size 和 num_heads * head_dim",
            "把 q/k/v 的维度顺序从 (B,H,S,D) 写成其他顺序",
            "忘记 K 需要转置后再与 Q 相乘",
            "causal mask 方向写反",
            "把这个教学流水线误认为完整 LLM 推理服务",
        ),
        next_gate=(
            "完成本阶段后，你可以继续扩展 KV cache、采样循环、更多 fused kernel 或高性能并行 reduction。"
        ),
    ),
)


_STAGE_BY_ID = {stage.id: stage for stage in STAGES}


def iter_stages() -> Iterator[StageSpec]:
    return iter(STAGES)


def get_stage(stage_id: str) -> StageSpec:
    try:
        return _STAGE_BY_ID[stage_id]
    except KeyError as exc:
        available = ", ".join(_STAGE_BY_ID)
        raise ValueError(f"unknown stage id {stage_id!r}; available: {available}") from exc


__all__ = ["STAGES", "StageSpec", "get_stage", "iter_stages"]
