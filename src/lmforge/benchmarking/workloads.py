"""Reference FP32 training and attention benchmark workloads."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from ..nn.attention import MultiHeadSelfAttention
from ..nn.transformer import TransformerLM
from ..reproducibility import configure_reproducibility
from ..training.loss import clip_gradients, cross_entropy
from ..training.optimizer import AdamW
from .runner import measure_phases


def _validate_reference_fp32(device: str, precision: str, attention_backend: str) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Phase 2 baseline benchmarks require an available CUDA device")
    if precision != "float32":
        raise ValueError("P2-01 supports only float32")
    if attention_backend not in {"reference", "naive"}:
        raise ValueError("P2-01 supports only reference/naive attention")
    return target


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def training_configuration(
    *,
    batch_size: int,
    context_length: int,
    warmup: int,
    repetitions: int,
    precision: str,
    attention_backend: str,
) -> dict[str, Any]:
    return {
        "operation": "training_step",
        "batch_size": batch_size,
        "context_length": context_length,
        "precision": precision,
        "dtype": "torch.float32",
        "attention_backend": attention_backend,
        "warmup": warmup,
        "repetitions": repetitions,
        "cold_start_iterations": 1,
        "input_batch": "pre_generated_device_resident",
    }


def run_training_workload(
    *,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: str,
    precision: str,
    attention_backend: str,
    seed: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    target = _validate_reference_fp32(device, precision, attention_backend)
    configure_reproducibility(seed, deterministic=False)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=target,
        dtype=torch.float32,
    )
    model.train()
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    generator = torch.Generator(device=target).manual_seed(seed)
    inputs = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        device=target,
        generator=generator,
    )
    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        device=target,
        generator=generator,
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        clip_gradients(model.parameters(), 1.0)
        optimizer.step()

    return {
        "model_parameters": _parameter_count(model),
        "measurement": measure_phases(
            step,
            device=target,
            warmup=warmup,
            repetitions=repetitions,
            tokens_per_step=batch_size * context_length,
        ),
    }


def attention_configuration(
    *,
    operation: str,
    batch_size: int,
    context_length: int,
    warmup: int,
    repetitions: int,
    precision: str,
    attention_backend: str,
) -> dict[str, Any]:
    return {
        "operation": operation,
        "batch_size": batch_size,
        "context_length": context_length,
        "precision": precision,
        "dtype": "torch.float32",
        "attention_backend": attention_backend,
        "warmup": warmup,
        "repetitions": repetitions,
        "cold_start_iterations": 1,
        "input_batch": "pre_generated_device_resident",
    }


def run_attention_workload(
    *,
    operation: str,
    batch_size: int,
    context_length: int,
    d_model: int,
    num_heads: int,
    rope_theta: float,
    device: str,
    precision: str,
    attention_backend: str,
    seed: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    if operation not in {"forward_only", "forward_backward"}:
        raise ValueError(f"unknown attention operation: {operation}")
    target = _validate_reference_fp32(device, precision, attention_backend)
    configure_reproducibility(seed, deterministic=False)
    attention = MultiHeadSelfAttention(
        d_model,
        num_heads,
        use_rope=True,
        max_seq=context_length,
        theta=rope_theta,
        device=target,
        dtype=torch.float32,
    )
    attention.train(operation == "forward_backward")
    generator = torch.Generator(device=target).manual_seed(seed)
    inputs = torch.randn(
        batch_size,
        context_length,
        d_model,
        device=target,
        dtype=torch.float32,
        generator=generator,
        requires_grad=operation == "forward_backward",
    )
    positions = torch.arange(context_length, device=target).expand(batch_size, -1)

    if operation == "forward_only":

        def step() -> None:
            with torch.no_grad():
                attention(inputs, token_positions=positions)
    else:

        def step() -> None:
            attention.zero_grad(set_to_none=True)
            inputs.grad = None
            output = attention(inputs, token_positions=positions)
            output.sum().backward()

    return {
        "model_parameters": _parameter_count(attention),
        "measurement": measure_phases(
            step,
            device=target,
            warmup=warmup,
            repetitions=repetitions,
            tokens_per_step=batch_size * context_length,
        ),
    }


def profile_training_workload(
    *,
    trace_path: Path,
    operator_path: Path,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: str,
    seed: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    target = _validate_reference_fp32(device, "float32", "reference")
    configure_reproducibility(seed, deterministic=False)
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=target,
        dtype=torch.float32,
    )
    model.train()
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    generator = torch.Generator(device=target).manual_seed(seed)
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=target, generator=generator)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=target, generator=generator)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        clip_gradients(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(target)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for _ in range(repetitions):
            step()
            profiler.step()
        # Keep the synchronization inside the profiler context so CUDA kernels
        # are associated with their launching operators in key_averages().
        torch.cuda.synchronize(target)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    operator_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))

    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append(
            {
                "operator": event.key,
                "calls": event.count,
                "input_shapes": repr(event.input_shapes),
                "self_cpu_time_total_us": event.self_cpu_time_total,
                "cpu_time_total_us": event.cpu_time_total,
                "self_cuda_time_total_us": getattr(event, "self_device_time_total", 0.0),
                "cuda_time_total_us": getattr(event, "device_time_total", 0.0),
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "self_cuda_memory_bytes": getattr(event, "self_device_memory_usage", 0),
            }
        )
    cuda_operator_timing_available = any(float(row["self_cuda_time_total_us"]) > 0 for row in rows)
    sort_metric = "self_cuda_time_total_us" if cuda_operator_timing_available else "cpu_time_total_us"
    rows.sort(key=lambda row: float(row[sort_metric]), reverse=True)
    fieldnames = list(rows[0]) if rows else ["operator"]
    with operator_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "status": "ok",
        "formal_timing": False,
        "model_parameters": _parameter_count(model),
        "warmup": warmup,
        "profiled_steps": repetitions,
        "cuda_operator_timing_available": cuda_operator_timing_available,
        "operator_sort_metric": sort_metric,
        "trace": str(trace_path.resolve()),
        "operator_summary": str(operator_path.resolve()),
    }
