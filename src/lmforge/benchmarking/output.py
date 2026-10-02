"""Stable JSON and flat CSV benchmark output."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _get(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for name in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    return value


def _csv_row(report: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    configuration = result.get("configuration", {})
    steady = _get(result, "measurement", "steady_state") or {}
    failure = result.get("failure") or {}
    return {
        "benchmark": report.get("benchmark"),
        "git_sha": _get(report, "git", "sha"),
        "git_dirty": _get(report, "git", "dirty"),
        "seed": report.get("seed"),
        "status": result.get("status"),
        "operation": configuration.get("operation"),
        "batch_size": configuration.get("batch_size"),
        "context_length": configuration.get("context_length"),
        "precision": configuration.get("precision"),
        "attention_backend": configuration.get("attention_backend"),
        "warmup": configuration.get("warmup"),
        "repetitions": configuration.get("repetitions"),
        "model_parameters": result.get("model_parameters"),
        "wall_step_seconds_median": _get(steady, "wall_time_seconds", "median"),
        "cuda_step_seconds_median": _get(steady, "cuda_event_time_seconds", "median"),
        "wall_tokens_per_second_median": _get(steady, "tokens_per_second_wall", "median"),
        "cuda_tokens_per_second_median": _get(steady, "tokens_per_second_cuda", "median"),
        "peak_allocated_bytes": _get(steady, "peak_memory", "allocated_bytes"),
        "peak_reserved_bytes": _get(steady, "peak_memory", "reserved_bytes"),
        "failure_type": failure.get("type"),
        "failure_message": failure.get("message"),
    }


def write_report(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = [_csv_row(report, result) for result in report.get("results", [])]
    fieldnames = list(_csv_row(report, {}).keys())
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
