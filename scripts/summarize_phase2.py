#!/usr/bin/env python3
"""Aggregate Phase 2 raw runs and render publication tables and SVG charts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lmforge.benchmarking.analysis import model_flops_utilization, theoretical_training_flops
from lmforge.benchmarking.phase2 import aggregate_records

FP32_PEAK_FLOPS = 2_048 * 1.740e9 * 2
BF16_DENSE_TENSOR_PEAK_FLOPS = FP32_PEAK_FLOPS * 4
MODEL = {"vocab_size": 8192, "d_model": 256, "num_layers": 4, "d_ff": 768}
METRICS = (
    "step_seconds",
    "tokens_per_second",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "cold_seconds",
    "step_flops",
    "mfu",
)
GROUPS = (
    "stage",
    "variant",
    "operation",
    "context_length",
    "batch_size",
    "precision",
    "attention_backend",
    "compile_model",
)


def _get(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _peak(precision: str) -> float:
    return BF16_DENSE_TENSOR_PEAK_FLOPS if precision == "bfloat16" else FP32_PEAK_FLOPS


def _micro_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("*/*/run-*/benchmark.json")):
        stage, variant, run_name = path.relative_to(root).parts[:3]
        report = json.loads(path.read_text(encoding="utf-8"))
        for result in report["results"]:
            config = result["configuration"]
            steady = _get(result, "measurement", "steady_state") or {}
            step_seconds = _get(steady, "cuda_event_time_seconds", "median")
            precision = str(config["precision"])
            step_flops = None
            mfu = None
            if config.get("operation") == "training_step":
                step_flops = theoretical_training_flops(
                    batch_size=int(config["batch_size"]),
                    context_length=int(config["context_length"]),
                    **MODEL,
                )
                if result["status"] == "ok" and isinstance(step_seconds, (int, float)):
                    mfu = model_flops_utilization(
                        step_flops=step_flops,
                        step_seconds=float(step_seconds),
                        peak_flops_per_second=_peak(precision),
                    )
            records.append(
                {
                    "stage": stage,
                    "variant": variant,
                    "run": int(run_name.split("-")[-1]),
                    "status": result["status"],
                    "failure_type": _get(result, "failure", "type"),
                    "operation": config.get("operation"),
                    "context_length": config.get("context_length"),
                    "batch_size": config.get("batch_size"),
                    "precision": precision,
                    "attention_backend": config.get("attention_backend"),
                    "compile_model": bool(config.get("compile_model", False)),
                    "step_seconds": step_seconds,
                    "tokens_per_second": _get(steady, "tokens_per_second_cuda", "median"),
                    "peak_allocated_bytes": _get(steady, "peak_memory", "allocated_bytes"),
                    "peak_reserved_bytes": _get(steady, "peak_memory", "reserved_bytes"),
                    "cold_seconds": _get(result, "measurement", "cold_start", "cuda_event_seconds"),
                    "step_flops": step_flops,
                    "mfu": mfu,
                }
            )
    return records


def _long_records(root: Path) -> list[dict[str, Any]]:
    records = []
    preferred = root / "long-valid"
    base = preferred if any(preferred.glob("*/run-*/metrics.jsonl")) else root / "long"
    for path in sorted(base.glob("*/run-*/metrics.jsonl")):
        variant, run_name = path.relative_to(base).parts[:2]
        metrics = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        steady = metrics[10:]
        step_seconds = sum(float(row["train_seconds"]) for row in steady) / len(steady)
        tokens_per_second = sum(float(row["tokens_per_second"]) for row in steady) / len(steady)
        precision = str(config["precision"])
        step_flops = theoretical_training_flops(
            batch_size=int(config["batch_size"]) * int(config["grad_accum_steps"]),
            context_length=int(config["model"]["context_length"]),
            **MODEL,
        )
        final = metrics[-1]
        status = "ok" if len(metrics) == 1000 and final["step"] == 1000 else "incomplete"
        records.append(
            {
                "stage": "long",
                "variant": variant,
                "run": int(run_name.split("-")[-1]),
                "status": status,
                "operation": "training_1000_steps",
                "context_length": config["model"]["context_length"],
                "batch_size": config["batch_size"],
                "gradient_accumulation": config["grad_accum_steps"],
                "precision": precision,
                "attention_backend": config["model"].get("attention_backend", "reference"),
                "compile_model": bool(config.get("compile_model", False)),
                "steps": len(metrics),
                "step_seconds": step_seconds,
                "tokens_per_second": tokens_per_second,
                "cold_seconds": float(metrics[0]["train_seconds"]),
                "step_flops": step_flops,
                "mfu": model_flops_utilization(
                    step_flops=step_flops,
                    step_seconds=step_seconds,
                    peak_flops_per_second=_peak(precision),
                ),
                "final_train_loss": float(final["train_loss"]),
                "final_validation_loss": float(final["val_loss"]),
                "skipped_steps": sum(bool(row["optimizer_step_skipped"]) for row in metrics),
            }
        )
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for field, value in row.items()}
            )


def _svg_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
) -> str:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 55, 75
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    points = [point for values in series.values() for point in values]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    y_min = min(0.0, y_min)
    y_max *= 1.08

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + (x - x_min) / max(x_max - x_min, 1) * (width - left - right)
        py = top + (y_max - y) / max(y_max - y_min, 1e-12) * (height - top - bottom)
        return px, py

    content = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">{title}</text>',
    ]
    for index in range(6):
        fraction = index / 5
        y = top + fraction * (height - top - bottom)
        value = y_max - fraction * (y_max - y_min)
        content.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        content.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.2f}</text>')
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, values))
        content.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in map(project, values):
            content.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        legend_y = 55 + index * 20
        content.append(f'<line x1="650" y1="{legend_y}" x2="675" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        content.append(f'<text x="682" y="{legend_y+4}" font-size="12" font-family="sans-serif">{name}</text>')
    for value in sorted(set(x_values)):
        x, _ = project((value, y_min))
        content.append(f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-size="12">{value:g}</text>')
    content.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14">{x_label}</text>',
            f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {height/2})">{y_label}</text>',
            "</svg>",
            "",
        ]
    )
    return "\n".join(content)


def _charts(output: Path, summaries: list[dict[str, Any]]) -> None:
    control = [row for row in summaries if row["stage"] == "control" and row["readme_eligible"]]
    throughput: dict[str, list[tuple[float, float]]] = defaultdict(list)
    memory: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in control:
        throughput[row["variant"]].append((row["context_length"], row["tokens_per_second_mean"] / 1000))
        memory[row["variant"]].append((row["context_length"], row["peak_allocated_bytes_mean"] / 2**20))
    (output / "context-throughput.svg").write_text(
        _svg_chart(title="Batch-1 Throughput vs Context", x_label="context length", y_label="thousand tokens/s", series=dict(throughput)),
        encoding="utf-8",
    )
    (output / "context-memory.svg").write_text(
        _svg_chart(title="Batch-1 Peak Allocated Memory", x_label="context length", y_label="MiB", series=dict(memory)),
        encoding="utf-8",
    )
    scaling = [
        row for row in summaries
        if row["stage"] == "capacity" and row["context_length"] == 512 and row["readme_eligible"]
    ]
    batch_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in scaling:
        batch_series[row["variant"]].append((row["batch_size"], row["tokens_per_second_mean"] / 1000))
    (output / "batch-scaling-c512.svg").write_text(
        _svg_chart(title="Context 512 Batch Scaling", x_label="batch size", y_label="thousand tokens/s", series=dict(batch_series)),
        encoding="utf-8",
    )
    attention = [row for row in summaries if row["stage"] == "attention" and row["readme_eligible"]]
    lookup = {(row["attention_backend"], row["operation"], row["context_length"]): row for row in attention}
    speedup: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for operation in ("forward_only", "forward_backward"):
        for context in (128, 256, 512, 1024, 2048):
            ref = lookup[("reference", operation, context)]["step_seconds_mean"]
            sdpa = lookup[("sdpa", operation, context)]["step_seconds_mean"]
            speedup[operation].append((context, ref / sdpa))
    (output / "attention-speedup.svg").write_text(
        _svg_chart(title="SDPA Speedup over Reference Attention", x_label="context length", y_label="speedup (x)", series=dict(speedup)),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    micro = _micro_records(args.input_dir)
    summaries = aggregate_records(micro, group_fields=GROUPS, metric_fields=METRICS, required_runs=3)
    long = _long_records(args.input_dir)
    long_summaries = aggregate_records(
        long,
        group_fields=("stage", "variant", "operation", "context_length", "batch_size", "precision", "attention_backend", "compile_model"),
        metric_fields=("step_seconds", "tokens_per_second", "cold_seconds", "step_flops", "mfu", "final_train_loss", "final_validation_loss", "skipped_steps"),
        required_runs=3,
    )
    metadata = {
        "benchmark_commit": "fd11e1c3930e1e554d4ce77c26ebfc0c3976ec1c",
        "aggregation": "mean and sample standard deviation across three independent-run steady-state medians",
        "flops_formula": "3 * (layers * (8*B*T*d^2 + 4*B*T^2*d + 6*B*T*d*d_ff) + 2*B*T*d*vocab)",
        "flops_exclusions": "embedding lookup, normalization, activation, softmax, loss, clipping, optimizer elementwise work",
        "hardware_peaks": {
            "fp32_flops_per_second": FP32_PEAK_FLOPS,
            "bf16_dense_tensor_flops_per_second": BF16_DENSE_TENSOR_PEAK_FLOPS,
            "assumption": "2048 CUDA cores, 1.740 GHz official maximum boost; dense GA10x BF16 Tensor Core peak is 4x FP32; no sparsity",
        },
    }
    payload = {"metadata": metadata, "micro_summary": summaries, "long_summary": long_summaries}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "raw-micro-runs.csv", micro)
    _write_csv(output / "micro-summary.csv", summaries)
    _write_csv(output / "raw-long-runs.csv", long)
    _write_csv(output / "long-summary.csv", long_summaries)
    _charts(output, summaries)
    if not all(row["readme_eligible"] for row in summaries if row["status_counts"].get("ok")):
        raise SystemExit("one or more successful microbenchmark groups lack three clean runs")
    if not all(row["readme_eligible"] for row in long_summaries):
        raise SystemExit("long-training publication gate failed")
    print(json.dumps({"micro_groups": len(summaries), "long_groups": len(long_summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
