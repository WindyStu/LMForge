"""Training and attention benchmark contracts."""

from __future__ import annotations

import csv
import json
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
                },
                "model_parameters": 123,
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


@pytest.mark.parametrize("name", ["benchmark_training.py", "benchmark_attention.py"])
def test_benchmark_scripts_are_independent_cli_entrypoints(name: str) -> None:
    script = Path(__file__).parents[1] / "scripts" / name
    assert script.is_file()
    source = script.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in source
