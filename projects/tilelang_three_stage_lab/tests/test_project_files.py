from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tilelab.common import BenchmarkResult, write_benchmark_csv
from tilelab.stages import STAGES, get_stage


def test_learning_entry_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "README.md",
        "docs/learning_path.md",
        "docs/lab_index.md",
        "docs/lab_answer_key.md",
        "docs/stage_map.md",
        "docs/tilelang_semantics.md",
        "docs/tensor_shapes.md",
        "docs/shape_walkthroughs.md",
        "docs/debug_checklist.md",
        "docs/error_gallery.md",
        "docs/performance_guide.md",
        "docs/optimization_log.md",
        "docs/reduction_optimization.md",
        "benchmarks/bench_optimization.py",
        "benchmarks/bench_reductions.py",
        "labs/README.md",
        "labs/01_basic_kernel_debug/README.md",
        "labs/02_gemm_tile_shape/README.md",
        "labs/03_reduction_optimization/README.md",
        "labs/04_decoder_shape_trace/README.md",
        "labs/05_benchmark_reading/README.md",
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
        "tilelab/stages.py",
        "reports/stage01_basic_template.md",
        "reports/stage02_advanced_template.md",
        "reports/stage03_decoder_template.md",
        "reports/perf_optimization_template.csv",
        "reports/reduction_optimization_template.md",
    ]
    missing = [path for path in expected if not (root / path).exists()]
    assert missing == []


def test_stage_metadata_is_complete() -> None:
    assert [stage.id for stage in STAGES] == ["basic", "advanced", "decoder"]
    assert [stage.order for stage in STAGES] == [1, 2, 3]

    for stage in STAGES:
        assert stage.name
        assert stage.path
        assert stage.level
        assert stage.goal
        assert stage.prerequisites
        assert stage.core_concepts
        assert stage.kernels
        assert stage.run_commands
        assert stage.acceptance_checks
        assert stage.deliverables
        assert stage.common_mistakes
        assert stage.next_gate

    assert get_stage("basic").path == "01_basic_kernels"
    assert get_stage("advanced").path == "02_advanced_ops"
    assert get_stage("decoder").path == "03_decoder_block_pipeline"


def test_stage_docs_have_learning_contract_sections() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = {
        "basic": root / "docs/stages/01_basic_kernels.md",
        "advanced": root / "docs/stages/02_advanced_ops.md",
        "decoder": root / "docs/stages/03_decoder_block_pipeline.md",
    }
    required_sections = [
        "## 阶段定位",
        "## 细化学习目标",
        "## 必会概念",
        "## 逐步任务",
        "## 必须运行",
        "## 实验记录要求",
        "## 验收问题",
        "## 常见卡点",
    ]

    for stage in STAGES:
        text = docs[stage.id].read_text(encoding="utf-8")
        for section in required_sections:
            assert section in text, f"{docs[stage.id]} missing {section}"
        for command in stage.run_commands[:3]:
            assert command in text


def test_stage_readmes_point_to_detailed_docs() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_docs = {
        "basic": "../docs/stages/01_basic_kernels.md",
        "advanced": "../docs/stages/02_advanced_ops.md",
        "decoder": "../docs/stages/03_decoder_block_pipeline.md",
    }
    for stage in STAGES:
        text = (root / stage.path / "README.md").read_text(encoding="utf-8")
        assert expected_docs[stage.id] in text
        assert "## 细化学习目标" in text
        assert "## 验收标准" in text


def test_lab_docs_are_actionable() -> None:
    root = Path(__file__).resolve().parents[1]
    labs = [
        "labs/01_basic_kernel_debug/README.md",
        "labs/02_gemm_tile_shape/README.md",
        "labs/03_reduction_optimization/README.md",
        "labs/04_decoder_shape_trace/README.md",
        "labs/05_benchmark_reading/README.md",
    ]
    for relpath in labs:
        text = (root / relpath).read_text(encoding="utf-8")
        assert "## 目标" in text
        assert "## 要读的代码" in text
        assert "## 运行命令" in text
        assert "## 动手改法" in text
        assert "## 自测问题" in text


def test_lab_answer_key_mentions_all_labs() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/lab_answer_key.md").read_text(encoding="utf-8")
    assert "Basic Kernel Debug" in text
    assert "GEMM Tile Shape" in text
    assert "Reduction Optimization" in text
    assert "Decoder Shape Trace" in text
    assert "Benchmark Reading" in text


def test_check_project_stage_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_project.py",
            "--list-stages",
            "--stage",
            "basic",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "[basic]" in result.stdout
    assert "== Stage Checks ==" in result.stdout
    assert "python3 scripts/check_project.py --stage basic" in result.stdout


def test_learning_lab_cli_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    list_result = subprocess.run(
        [sys.executable, "scripts/run_lab.py", "--list"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert list_result.returncode == 0, list_result.stderr
    assert "Reduction Optimization" in list_result.stdout

    run_result = subprocess.run(
        [sys.executable, "scripts/run_lab.py", "--lab", "all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert "full labs ok" in run_result.stdout

    error_result = subprocess.run(
        [sys.executable, "scripts/demo_common_errors.py", "--case", "all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert error_result.returncode == 0, error_result.stderr
    assert "error demos completed" in error_result.stdout
    assert "caught:" in error_result.stdout


def test_benchmark_csv_helper_writes_expected_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "bench.csv"
    write_benchmark_csv(
        csv_path,
        [
            BenchmarkResult(
                "tilelang_scale",
                0.123456,
                1,
                2,
                suite="optimization",
                shape="M=128,N=128",
                dtype="float16",
                backend="tilelang",
                notes="smoke",
            )
        ],
    )
    text = csv_path.read_text(encoding="utf-8")
    assert "suite,name,backend,shape,dtype,latency_ms,warmup,repeat,notes" in text
    assert "optimization,tilelang_scale,tilelang" in text


def test_benchmark_csv_summary_cli_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "bench.csv"
    write_benchmark_csv(
        csv_path,
        [
            BenchmarkResult(
                "torch_row_sum",
                0.2,
                1,
                2,
                suite="reductions",
                shape="M=2,N=4,block_n=4",
                dtype="float16",
                backend="torch",
            ),
            BenchmarkResult(
                "tilelang_row_sum_serial",
                0.3,
                1,
                2,
                suite="reductions",
                shape="M=2,N=4,block_n=4",
                dtype="float16",
                backend="tilelang",
            ),
            BenchmarkResult(
                "tilelang_row_sum_parallel",
                0.1,
                1,
                2,
                suite="reductions",
                shape="M=2,N=4,block_n=4",
                dtype="float16",
                backend="tilelang",
            ),
        ],
    )
    result = subprocess.run(
        [sys.executable, "scripts/summarize_bench_csv.py", "--csv", str(csv_path)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "fastest" in result.stdout
    assert "serial vs parallel" in result.stdout
