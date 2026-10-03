"""Frozen Phase 2 publication protocol."""

from __future__ import annotations

from typing import Any


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
