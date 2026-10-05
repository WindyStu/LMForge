"""Training and attention benchmark contracts."""

from __future__ import annotations

import csv
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch


def test_measurement_separates_cold_warmup_and_steady_state() -> None:
    from lmforge.benchmarking.runner import measure_phases

    calls: list[str] = []

    def step() -> None:
        calls.append("step")

    result = measure_phases(
        step,
        device="cpu",
        warmup=2,
        repetitions=3,
        tokens_per_step=8,
    )

    assert calls == ["step"] * 6
    assert result["cold_start"]["phase"] == "cold_start"
    assert result["warmup"]["iterations"] == 2
    assert len(result["steady_state"]["iterations"]) == 3
    assert result["steady_state"]["tokens_per_step"] == 8
    assert result["steady_state"]["wall_time_seconds"]["median"] > 0
    assert result["steady_state"]["cuda_event_time_seconds"] is None


def test_cuda_measurement_resets_peaks_only_at_phase_boundaries(monkeypatch) -> None:
    from lmforge.benchmarking import runner

    events: list[str] = []

    monkeypatch.setattr(runner, "_synchronize", lambda _device: events.append("sync"))
    monkeypatch.setattr(runner, "_reset_peak_memory_stats", lambda _device: events.append("reset"))
    monkeypatch.setattr(
        runner,
        "_peak_memory",
        lambda _device: {"allocated_bytes": 10, "reserved_bytes": 20},
    )
    monkeypatch.setattr(
        runner,
        "_measure_iteration",
        lambda step, _device: (step(), {"wall_seconds": 1.0, "cuda_event_seconds": 0.5})[1],
    )

    runner.measure_phases(
        lambda: events.append("step"),
        device="cuda",
        warmup=2,
        repetitions=2,
        tokens_per_step=4,
    )

    assert events.count("reset") == 2
    assert events == [
        "sync",
        "reset",
        "step",
        "sync",
        "step",
        "step",
        "sync",
        "reset",
        "step",
        "step",
        "sync",
    ]


def test_report_writes_json_and_flat_csv_with_environment_metadata(tmp_path: Path) -> None:
    from lmforge.benchmarking.output import write_report

    report = {
        "format": "lmforge-benchmark-v1",
        "benchmark": "training",
        "command": ["python", "benchmark_training.py"],
        "git": {"sha": "abc", "dirty": False},
        "environment": {"torch": "2.6", "cuda": "12.4", "gpus": [{"name": "GPU"}]},
        "results": [
            {
                "status": "ok",
                "configuration": {
                    "batch_size": 1,
                    "context_length": 128,
                    "precision": "float32",
                    "attention_backend": "reference",
                    "compile_model": True,
                },
                "model_parameters": 123,
                "compile_diagnostics": {
                    "graph_break_count": 2,
                    "fallback_detected": True,
                },
                "measurement": {
                    "steady_state": {
                        "wall_time_seconds": {"median": 0.5},
                        "cuda_event_time_seconds": {"median": 0.4},
                        "tokens_per_second_wall": {"median": 256.0},
                        "tokens_per_second_cuda": {"median": 320.0},
                        "peak_memory": {"allocated_bytes": 10, "reserved_bytes": 20},
                    }
                },
            }
        ],
    }

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    write_report(report, json_path=json_path, csv_path=csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["git_sha"] == "abc"
    assert rows[0]["batch_size"] == "1"
    assert rows[0]["wall_tokens_per_second_median"] == "256.0"
    assert rows[0]["torch_version"] == "2.6"
    assert rows[0]["cuda_version"] == "12.4"
    assert rows[0]["gpu_name"] == "GPU"
    assert rows[0]["compile_model"] == "True"
    assert rows[0]["graph_break_count"] == "2"
    assert rows[0]["compile_fallback_detected"] == "True"
    assert json.loads(rows[0]["command_json"]) == [
        "python",
        "benchmark_training.py",
    ]


def test_bfloat16_configuration_reports_actual_dtype() -> None:
    from lmforge.benchmarking.workloads import training_configuration

    configuration = training_configuration(
        batch_size=2,
        context_length=128,
        warmup=3,
        repetitions=4,
        precision="bfloat16",
        attention_backend="sdpa",
    )

    assert configuration["precision"] == "bfloat16"
    assert configuration["dtype"] == "torch.bfloat16"


def test_oom_result_records_failure_and_uses_cleanup_only_after_oom(monkeypatch) -> None:
    from lmforge.benchmarking import runner

    cleanups: list[str] = []
    monkeypatch.setattr(runner, "cleanup_after_oom", lambda _device: cleanups.append("cleanup"))

    def raise_oom():
        raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")

    result = runner.run_configuration(raise_oom, device="cuda")

    assert result["status"] == "oom"
    assert "out of memory" in result["failure"]["message"].lower()
    assert cleanups == ["cleanup"]


def test_non_oom_failure_is_recorded_without_empty_cache(monkeypatch) -> None:
    from lmforge.benchmarking import runner

    cleanups: list[str] = []
    monkeypatch.setattr(runner, "cleanup_after_oom", lambda _device: cleanups.append("cleanup"))

    def fail():
        raise ValueError("bad configuration")

    result = runner.run_configuration(fail, device="cuda")

    assert result["status"] == "failed"
    assert result["failure"]["type"] == "ValueError"
    assert cleanups == []


def test_oom_cleanup_failure_does_not_hide_original_oom(monkeypatch) -> None:
    from lmforge.benchmarking import runner

    def cleanup_fails(_device: str) -> None:
        raise RuntimeError("CUDA remained out of memory during empty_cache")

    monkeypatch.setattr(runner, "cleanup_after_oom", cleanup_fails)

    def raise_oom() -> None:
        raise torch.cuda.OutOfMemoryError("original allocation failed")

    result = runner.run_configuration(raise_oom, device="cuda")

    assert result["status"] == "oom"
    assert result["failure"]["type"] == "OutOfMemoryError"
    assert result["failure"]["message"] == "original allocation failed"


def test_wsl_oversubscription_is_recorded_as_a_memory_capacity_boundary() -> None:
    from lmforge.benchmarking.runner import enforce_device_memory_capacity

    result = {
        "status": "ok",
        "measurement": {"steady_state": {"peak_memory": {"allocated_bytes": 9, "reserved_bytes": 12}}},
    }

    bounded = enforce_device_memory_capacity(result, total_memory_bytes=10)

    assert bounded["status"] == "memory_capacity_exceeded"
    assert bounded["failure"]["type"] == "DeviceMemoryCapacityExceeded"
    assert bounded["failure"]["total_memory_bytes"] == 10
    assert bounded["failure"]["peak_reserved_bytes"] == 12


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_training.py",
        "benchmark_attention.py",
        "benchmark_phase2.py",
        "summarize_phase2.py",
    ],
)
def test_benchmark_scripts_are_independent_cli_entrypoints(name: str) -> None:
    script = Path(__file__).parents[1] / "scripts" / name
    assert script.is_file()
    source = script.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in source


def test_training_flops_scale_with_batch_and_include_quadratic_attention() -> None:
    from lmforge.benchmarking.analysis import theoretical_training_flops

    short = theoretical_training_flops(
        batch_size=1,
        context_length=128,
        vocab_size=8192,
        d_model=256,
        num_layers=4,
        d_ff=768,
    )
    long = theoretical_training_flops(
        batch_size=1,
        context_length=256,
        vocab_size=8192,
        d_model=256,
        num_layers=4,
        d_ff=768,
    )
    doubled_batch = theoretical_training_flops(
        batch_size=2,
        context_length=128,
        vocab_size=8192,
        d_model=256,
        num_layers=4,
        d_ff=768,
    )

    assert doubled_batch == 2 * short
    assert long > 2 * short


def test_mfu_requires_explicit_peak_and_uses_step_flops() -> None:
    from lmforge.benchmarking.analysis import model_flops_utilization

    assert model_flops_utilization(step_flops=2_000, step_seconds=0.5, peak_flops_per_second=10_000) == 0.4
    with pytest.raises(ValueError, match="peak"):
        model_flops_utilization(step_flops=2_000, step_seconds=0.5, peak_flops_per_second=0)


def test_independent_run_summary_reports_sample_variation() -> None:
    from lmforge.benchmarking.analysis import summarize_independent_runs

    summary = summarize_independent_runs([10.0, 12.0, 14.0])

    assert summary == {
        "count": 3,
        "mean": 12.0,
        "stdev": 2.0,
        "min": 10.0,
        "max": 14.0,
    }


def test_phase2_protocol_keeps_control_capacity_attention_and_long_training() -> None:
    from lmforge.benchmarking.phase2 import formal_protocol

    protocol = formal_protocol(independent_runs=3, long_training_steps=1000)

    assert protocol["contexts"] == [128, 256, 512, 1024, 2048]
    assert protocol["warmup"] == 5
    assert protocol["steady_repetitions"] == 10
    assert protocol["independent_runs"] == 3
    assert len(protocol["control_variants"]) == 5
    assert {item["name"] for item in protocol["capacity_variants"]} == {
        "reference-fp32-eager",
        "sdpa-bf16-eager",
    }
    assert {item["attention_backend"] for item in protocol["attention_variants"]} == {
        "reference",
        "sdpa",
    }
    assert protocol["long_training"]["steps"] == 1000
    assert protocol["long_training"]["context_length"] == 256
    assert protocol["long_training"]["batch_size"] == 4
    assert protocol["long_training"]["gradient_accumulation"] == 8
    assert not any(
        forbidden in repr(protocol).lower()
        for forbidden in ("flashattention", "triton", "kv cache")
    )


def test_publication_gate_requires_three_successful_independent_runs() -> None:
    from lmforge.benchmarking.phase2 import publication_eligibility

    assert publication_eligibility(["ok", "ok", "ok"], required_runs=3) is True
    assert publication_eligibility(["ok", "ok"], required_runs=3) is False
    assert publication_eligibility(["ok", "ok", "oom"], required_runs=3) is False


def test_aggregate_records_keeps_failures_and_gates_incomplete_groups() -> None:
    from lmforge.benchmarking.phase2 import aggregate_records

    records = [
        {"variant": "a", "context": 128, "run": 1, "status": "ok", "throughput": 10.0},
        {"variant": "a", "context": 128, "run": 2, "status": "ok", "throughput": 12.0},
        {"variant": "a", "context": 128, "run": 3, "status": "ok", "throughput": 14.0},
        {"variant": "b", "context": 128, "run": 1, "status": "ok", "throughput": 20.0},
        {"variant": "b", "context": 128, "run": 2, "status": "oom", "throughput": None},
        {"variant": "b", "context": 128, "run": 3, "status": "ok", "throughput": 22.0},
    ]

    summaries = aggregate_records(
        records,
        group_fields=("variant", "context"),
        metric_fields=("throughput",),
        required_runs=3,
    )

    assert summaries[0]["readme_eligible"] is True
    assert summaries[0]["throughput_mean"] == 12.0
    assert summaries[0]["throughput_stdev"] == 2.0
    assert summaries[1]["readme_eligible"] is False
    assert summaries[1]["status_counts"] == {"ok": 2, "oom": 1}
    assert summaries[1]["throughput_mean"] is None


def test_long_training_environment_enables_deterministic_cublas_and_isolated_compile_cache() -> None:
    from lmforge.benchmarking.phase2 import long_training_environment

    environment = long_training_environment(
        {"PATH": "/bin"},
        compile_model=True,
        cache_directory="/tmp/isolated-cache",
    )

    assert environment["PATH"] == "/bin"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/isolated-cache"


def test_phase2_orchestrator_runs_a_subprocess_and_records_logs(tmp_path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_phase2.py"
    namespace = runpy.run_path(str(script))
    destination = tmp_path / "run-01"

    result = namespace["_run"](
        {
            "stage": "control",
            "variant": "test",
            "run": 1,
            "output": str(destination),
            "command": [sys.executable, "-c", "print('ok')"],
        }
    )

    assert result["status"] == "ok"
    assert (destination / "stdout.log").read_text(encoding="utf-8") == "ok\n"
    assert json.loads((destination / "execution.json").read_text(encoding="utf-8"))[
        "returncode"
    ] == 0
