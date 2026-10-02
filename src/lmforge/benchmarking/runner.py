"""Cold-start, warmup, steady-state, and OOM measurement primitives."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

Step = Callable[[], None]


def _is_cuda(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def _synchronize(device: str | torch.device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(torch.device(device))


def _reset_peak_memory_stats(device: str | torch.device) -> None:
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_memory(device: str | torch.device) -> dict[str, int] | None:
    if not _is_cuda(device):
        return None
    target = torch.device(device)
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(target),
        "reserved_bytes": torch.cuda.max_memory_reserved(target),
    }


def _measure_iteration(step: Step, device: str | torch.device) -> dict[str, float | None]:
    target = torch.device(device)
    start_event = end_event = None
    if target.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    _synchronize(target)
    wall_start = time.perf_counter()
    if start_event is not None:
        start_event.record()
    step()
    if end_event is not None:
        end_event.record()
    _synchronize(target)
    wall_seconds = time.perf_counter() - wall_start
    cuda_seconds = None
    if start_event is not None and end_event is not None:
        cuda_seconds = start_event.elapsed_time(end_event) / 1000.0
    return {"wall_seconds": wall_seconds, "cuda_event_seconds": cuda_seconds}


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty measurement")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def measure_phases(
    step: Step,
    *,
    device: str | torch.device,
    warmup: int,
    repetitions: int,
    tokens_per_step: int,
) -> dict[str, Any]:
    """Measure exactly one cold step, unmeasured warmup, then steady-state steps."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if tokens_per_step < 1:
        raise ValueError("tokens_per_step must be positive")

    _synchronize(device)
    _reset_peak_memory_stats(device)
    cold = _measure_iteration(step, device)
    _synchronize(device)
    cold_peak = _peak_memory(device)

    for _ in range(warmup):
        step()

    _synchronize(device)
    _reset_peak_memory_stats(device)
    iterations = [_measure_iteration(step, device) for _ in range(repetitions)]
    _synchronize(device)
    steady_peak = _peak_memory(device)

    wall_values = [float(item["wall_seconds"]) for item in iterations]
    cuda_values = [float(item["cuda_event_seconds"]) for item in iterations if item["cuda_event_seconds"] is not None]
    wall_throughput = [tokens_per_step / value for value in wall_values]
    cuda_throughput = [tokens_per_step / value for value in cuda_values]
    return {
        "cold_start": {
            "phase": "cold_start",
            **cold,
            "tokens_per_step": tokens_per_step,
            "peak_memory": cold_peak,
        },
        "warmup": {"phase": "warmup", "iterations": warmup},
        "steady_state": {
            "phase": "steady_state",
            "iterations": iterations,
            "repetitions": repetitions,
            "tokens_per_step": tokens_per_step,
            "wall_time_seconds": _summary(wall_values),
            "cuda_event_time_seconds": _summary(cuda_values) if cuda_values else None,
            "tokens_per_second_wall": _summary(wall_throughput),
            "tokens_per_second_cuda": _summary(cuda_throughput) if cuda_throughput else None,
            "peak_memory": steady_peak,
        },
    }


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def cleanup_after_oom(device: str | torch.device) -> None:
    gc.collect()
    if _is_cuda(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_configuration(function: Callable[[], dict[str, Any] | None], *, device: str) -> dict[str, Any]:
    try:
        result = function()
    except Exception as error:  # noqa: BLE001 - failures belong in the benchmark report
        oom = is_oom(error)
        cleanup_failure = None
        if oom:
            try:
                cleanup_after_oom(device)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the original OOM
                cleanup_failure = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                }
        failure = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if cleanup_failure is not None:
            failure["cleanup_failure"] = cleanup_failure
        return {
            "status": "oom" if oom else "failed",
            "failure": failure,
        }
    return {"status": "ok", **({} if result is None else result)}
