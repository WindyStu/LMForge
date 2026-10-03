"""Pure analysis helpers for reproducible Phase 2 benchmark reports."""

from __future__ import annotations

import statistics


def theoretical_training_flops(
    *,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
) -> int:
    """Approximate dense forward+backward FLOPs for one optimizer step.

    Matrix multiplication counts use two FLOPs per multiply-accumulate and
    backward is approximated as twice the forward matrix-multiplication cost.
    Embedding lookup, normalization, activation, softmax, loss, clipping, and
    optimizer elementwise work are intentionally excluded.
    """

    values = (batch_size, context_length, vocab_size, d_model, num_layers, d_ff)
    if any(value < 1 for value in values):
        raise ValueError("model dimensions, batch size, and context length must be positive")
    tokens = batch_size * context_length
    per_layer_forward = (
        8 * tokens * d_model * d_model
        + 4 * batch_size * context_length * context_length * d_model
        + 6 * tokens * d_model * d_ff
    )
    output_projection_forward = 2 * tokens * d_model * vocab_size
    return 3 * (num_layers * per_layer_forward + output_projection_forward)


def model_flops_utilization(
    *,
    step_flops: int,
    step_seconds: float,
    peak_flops_per_second: float,
) -> float:
    """Return achieved model FLOPs divided by an explicit hardware peak."""

    if step_flops < 1:
        raise ValueError("step FLOPs must be positive")
    if step_seconds <= 0:
        raise ValueError("step time must be positive")
    if peak_flops_per_second <= 0:
        raise ValueError("peak FLOPs per second must be positive")
    return step_flops / step_seconds / peak_flops_per_second


def summarize_independent_runs(values: list[float]) -> dict[str, float | int]:
    """Summarize independent runs with sample standard deviation."""

    if not values:
        raise ValueError("cannot summarize zero independent runs")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }
