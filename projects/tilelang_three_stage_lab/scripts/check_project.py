from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES_FILE = ROOT / "tilelab" / "stages.py"


REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "tilelab/basic.py",
    "tilelab/advanced.py",
    "tilelab/decoder.py",
    "tilelab/stages.py",
    "tests/test_basic.py",
    "tests/test_advanced.py",
    "tests/test_decoder.py",
    "benchmarks/bench_basic.py",
    "benchmarks/bench_advanced.py",
    "benchmarks/bench_optimization.py",
    "benchmarks/bench_reductions.py",
    "benchmarks/bench_decoder_block.py",
    "labs/README.md",
    "labs/01_basic_kernel_debug/README.md",
    "labs/02_gemm_tile_shape/README.md",
    "labs/03_reduction_optimization/README.md",
    "labs/04_decoder_shape_trace/README.md",
    "labs/05_benchmark_reading/README.md",
    "docs/learning_path.md",
    "docs/lab_index.md",
    "docs/lab_answer_key.md",
    "docs/stage_map.md",
    "docs/shape_walkthroughs.md",
    "docs/error_gallery.md",
    "docs/performance_guide.md",
    "docs/optimization_log.md",
    "docs/reduction_optimization.md",
    "docs/stages/01_basic_kernels.md",
    "docs/stages/02_advanced_ops.md",
    "docs/stages/03_decoder_block_pipeline.md",
    "docs/kernels/vector_add.md",
    "docs/kernels/copy.md",
    "docs/kernels/axpy.md",
    "docs/kernels/row_sum.md",
    "docs/kernels/gemm.md",
    "docs/kernels/softmax.md",
    "docs/kernels/rmsnorm.md",
    "docs/kernels/gelu.md",
    "docs/kernels/decoder_block.md",
    "scripts/check_project.py",
    "scripts/run_lab.py",
    "scripts/demo_common_errors.py",
    "scripts/summarize_bench_csv.py",
    "reports/perf_optimization_template.csv",
    "reports/reduction_optimization_template.md",
]

STAGE_DOCS = {
    "basic": "docs/stages/01_basic_kernels.md",
    "advanced": "docs/stages/02_advanced_ops.md",
    "decoder": "docs/stages/03_decoder_block_pipeline.md",
}

STAGE_TESTS = {
    "basic": "tests/test_basic.py",
    "advanced": "tests/test_advanced.py",
    "decoder": "tests/test_decoder.py",
}

STAGE_BENCHMARKS = {
    "basic": "benchmarks/bench_basic.py",
    "advanced": "benchmarks/bench_advanced.py",
    "decoder": "benchmarks/bench_decoder_block.py",
}

STAGE_REPORTS = {
    "basic": "reports/stage01_basic_template.md",
    "advanced": "reports/stage02_advanced_template.md",
    "decoder": "reports/stage03_decoder_template.md",
}

STAGE_LABS = {
    "basic": [
        "labs/01_basic_kernel_debug/README.md",
        "labs/03_reduction_optimization/README.md",
    ],
    "advanced": [
        "labs/02_gemm_tile_shape/README.md",
        "labs/03_reduction_optimization/README.md",
        "labs/05_benchmark_reading/README.md",
    ],
    "decoder": [
        "labs/04_decoder_shape_trace/README.md",
        "labs/05_benchmark_reading/README.md",
    ],
}

KERNEL_DOCS = {
    "copy": "docs/kernels/copy.md",
    "vector_add": "docs/kernels/vector_add.md",
    "axpy": "docs/kernels/axpy.md",
    "row_sum": "docs/kernels/row_sum.md",
    "row_sum_parallel": "docs/reduction_optimization.md",
    "gemm": "docs/kernels/gemm.md",
    "row_softmax": "docs/kernels/softmax.md",
    "row_softmax_parallel": "docs/reduction_optimization.md",
    "softmax": "docs/kernels/softmax.md",
    "rmsnorm": "docs/kernels/rmsnorm.md",
    "rmsnorm_parallel": "docs/reduction_optimization.md",
    "gelu": "docs/kernels/gelu.md",
    "linear_bias_gelu": "docs/kernels/gelu.md",
    "causal_mask": "docs/kernels/decoder_block.md",
    "decoder_block": "docs/kernels/decoder_block.md",
    "decoder_block_optimized": "docs/reduction_optimization.md",
}


def load_stages():
    spec = importlib.util.spec_from_file_location("tilelab_stage_metadata", STAGES_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load stage metadata from {STAGES_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return tuple(module.iter_stages())


def print_environment() -> None:
    print("== Environment ==")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        print(f"torch cuda: {torch.version.cuda}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"gpu[{index}]: {torch.cuda.get_device_name(index)}")
    except Exception as exc:
        print(f"torch import failed: {exc}")

    try:
        import tilelang

        print(f"tilelang: {tilelang.__version__}")
    except Exception as exc:
        print(f"tilelang import failed: {exc}")


def print_files() -> bool:
    print("\n== Required Files ==")
    ok = True
    for relpath in REQUIRED_PATHS:
        exists = (ROOT / relpath).exists()
        print(f"{'ok  ' if exists else 'MISS'} {relpath}")
        ok = ok and exists
    return ok


def paths_for_stage(stage) -> list[str]:
    paths = [
        f"{stage.path}/README.md",
        STAGE_DOCS[stage.id],
        STAGE_TESTS[stage.id],
        STAGE_BENCHMARKS[stage.id],
        STAGE_REPORTS[stage.id],
    ]
    paths.extend(STAGE_LABS.get(stage.id, []))
    for kernel in stage.kernels:
        doc = KERNEL_DOCS.get(kernel)
        if doc is not None:
            paths.append(doc)
    return sorted(set(paths))


def print_stage_map(stages) -> None:
    print("\n== Stage Map ==")
    for stage in stages:
        print(f"[{stage.id}] {stage.name} ({stage.level})")
        print(f"  path: {stage.path}/")
        print(f"  goal: {stage.goal}")
        print(f"  kernels: {', '.join(stage.kernels)}")
        print(f"  next gate: {stage.next_gate}")


def print_stage_check(stages, selected_stage: str) -> bool:
    selected = stages if selected_stage == "all" else [stage for stage in stages if stage.id == selected_stage]
    ok = True
    print("\n== Stage Checks ==")
    for stage in selected:
        print(f"\n[{stage.id}] {stage.name}")
        print(f"goal: {stage.goal}")
        print("required files:")
        for relpath in paths_for_stage(stage):
            exists = (ROOT / relpath).exists()
            print(f"  {'ok  ' if exists else 'MISS'} {relpath}")
            ok = ok and exists
        print("commands:")
        for command in stage.run_commands:
            print(f"  {command}")
        print("deliverables:")
        for item in stage.deliverables:
            print(f"  - {item}")
        print("acceptance checks:")
        for item in stage.acceptance_checks:
            print(f"  - {item}")
    return ok


def print_commands(stages) -> None:
    print("\n== Recommended Commands ==")
    print("pytest -q")
    print("RUN_TILELANG_SMOKE=1 pytest -q -m tilelang")
    print("python3 scripts/run_lab.py --list")
    print("python3 scripts/run_lab.py --lab all")
    print("python3 scripts/demo_common_errors.py --case all")
    print("python3 scripts/summarize_bench_csv.py --csv /tmp/tilelang_reductions.csv")
    print("python3 -m benchmarks.bench_basic --run-tilelang --numel 1024 --warmup 1 --repeat 2")
    print(
        "python3 -m benchmarks.bench_optimization --run-tilelang "
        "--warmup 1 --repeat 2 --csv /tmp/tilelang_optimization.csv"
    )
    print(
        "python3 -m benchmarks.bench_reductions --run-tilelang --rows 128 --cols 128 "
        "--warmup 1 --repeat 2 --csv /tmp/tilelang_reductions.csv"
    )
    print(
        "python3 -m benchmarks.bench_advanced --run-tilelang --m 128 --n 128 --k 128 "
        "--warmup 1 --repeat 2 --csv /tmp/tilelang_advanced.csv"
    )
    print(
        "python3 -m benchmarks.bench_decoder_block --run-tilelang --compare-optimized "
        "--seq 128 --hidden 128 --heads 2 --ffn 128 --vocab 128 "
        "--warmup 1 --repeat 1 --csv /tmp/tilelang_decoder.csv"
    )
    print("\nStage-specific health checks:")
    for stage in stages:
        print(f"python3 scripts/check_project.py --stage {stage.id}")


def run_pytest() -> int:
    print("\n== Running pytest -q ==")
    return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)


def main() -> int:
    stages = load_stages()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-pytest", action="store_true", help="also run pytest -q")
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="print the three-stage learning map",
    )
    parser.add_argument(
        "--stage",
        choices=("all",) + tuple(stage.id for stage in stages),
        default="all",
        help="check files and learning tasks for one stage",
    )
    args = parser.parse_args()

    print(f"project: {ROOT}")
    print_environment()
    files_ok = print_files()
    if args.list_stages:
        print_stage_map(stages)
    stage_ok = print_stage_check(stages, args.stage)
    print_commands(stages)

    if args.run_pytest:
        return run_pytest()
    return 0 if files_ok and stage_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
