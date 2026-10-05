"""Frozen Phase 2 publication protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from .analysis import summarize_independent_runs


def _variant(name: str, precision: str, attention_backend: str, compile_model: bool) -> dict[str, Any]:
    return {
        "name": name,
        "precision": precision,
        "attention_backend": attention_backend,
        "compile_model": compile_model,
    }


def formal_protocol(*, independent_runs: int = 3, long_training_steps: int = 1000) -> dict[str, Any]:
    if independent_runs < 3:
        raise ValueError("formal publication protocol requires at least three independent runs")
    if long_training_steps < 100:
        raise ValueError("long training verification requires at least 100 optimizer steps")
    baseline = _variant("reference-fp32-eager", "float32", "reference", False)
    optimized_eager = _variant("sdpa-bf16-eager", "bfloat16", "sdpa", False)
    optimized_compile = _variant("sdpa-bf16-compile", "bfloat16", "sdpa", True)
    return {
        "seed": 42,
        "contexts": [128, 256, 512, 1024, 2048],
        "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "warmup": 5,
        "steady_repetitions": 10,
        "independent_runs": independent_runs,
        "model": {
            "vocab_size": 8192,
            "d_model": 256,
            "num_layers": 4,
            "num_heads": 4,
            "d_ff": 768,
            "rope_theta": 10000.0,
        },
        "control_variants": [
            baseline,
            _variant("sdpa-fp32-eager", "float32", "sdpa", False),
            _variant("reference-bf16-eager", "bfloat16", "reference", False),
            optimized_eager,
            optimized_compile,
        ],
        "capacity_variants": [baseline, optimized_eager],
        "attention_variants": [
            _variant("attention-reference-fp32", "float32", "reference", False),
            _variant("attention-sdpa-fp32", "float32", "sdpa", False),
        ],
        "long_training": {
            "steps": long_training_steps,
            "context_length": 256,
            "batch_size": 4,
            "gradient_accumulation": 8,
            "variants": [baseline, optimized_compile],
        },
    }


def publication_eligibility(statuses: list[str], *, required_runs: int = 3) -> bool:
    if required_runs < 1:
        raise ValueError("required runs must be positive")
    return len(statuses) >= required_runs and all(status == "ok" for status in statuses)


def long_training_environment(
    base: dict[str, str],
    *,
    compile_model: bool,
    cache_directory: str | None = None,
) -> dict[str, str]:
    """Return the deterministic CUDA environment for a formal long run."""

    if compile_model and cache_directory is None:
        raise ValueError("compiled long training requires an isolated cache directory")
    environment = dict(base)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if compile_model:
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_directory)
    return environment


def aggregate_records(
    records: Iterable[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
    required_runs: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate independent runs while preserving failed-group status counts."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(field) for field in group_fields)].append(record)
    summaries = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        group = grouped[key]
        statuses = [str(record.get("status")) for record in group]
        eligible = publication_eligibility(statuses, required_runs=required_runs)
        summary: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        summary["status_counts"] = dict(sorted(Counter(statuses).items()))
        summary["readme_eligible"] = eligible
        for field in metric_fields:
            values = [record.get(field) for record in group]
            if eligible and all(isinstance(value, (int, float)) for value in values):
                statistics = summarize_independent_runs([float(value) for value in values])
                for name, value in statistics.items():
                    summary[f"{field}_{name}"] = value
            else:
                for name in ("count", "mean", "stdev", "min", "max"):
                    summary[f"{field}_{name}"] = None
        summaries.append(summary)
    return summaries
