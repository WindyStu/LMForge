#!/usr/bin/env python3
"""Benchmark reference FP32 attention forward and forward-plus-backward."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lmforge.benchmarking.environment import benchmark_metadata
from lmforge.benchmarking.output import write_report
from lmforge.benchmarking.runner import (
    enforce_device_memory_capacity,
    run_configuration,
)
from lmforge.benchmarking.workloads import (
    attention_configuration,
    run_attention_workload,
)


def _positive_csv(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=_positive_csv, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--batch-sizes", type=_positive_csv, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument(
        "--operations", type=lambda value: value.split(","), default=["forward_only", "forward_backward"]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32",), default="float32")
    parser.add_argument("--attention-backend", choices=("reference", "naive"), default="reference")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--operation", help=argparse.SUPPRESS)
    parser.add_argument("--context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    return parser


def _configuration(args: argparse.Namespace, operation: str, context: int, batch_size: int) -> dict[str, object]:
    return attention_configuration(
        operation=operation,
        batch_size=batch_size,
        context_length=context,
        warmup=args.warmup,
        repetitions=args.repetitions,
        precision=args.precision,
        attention_backend=args.attention_backend,
    )


def _worker(args: argparse.Namespace) -> int:
    if args.operation is None or args.context is None or args.batch_size is None:
        raise SystemExit("--worker requires --operation, --context, and --batch-size")
    measured = run_configuration(
        lambda: run_attention_workload(
            operation=args.operation,
            batch_size=args.batch_size,
            context_length=args.context,
            d_model=args.d_model,
            num_heads=args.num_heads,
            rope_theta=args.rope_theta,
            device=args.device,
            precision=args.precision,
            attention_backend=args.attention_backend,
            seed=args.seed,
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        device=args.device,
    )
    print(
        json.dumps(
            {"configuration": _configuration(args, args.operation, args.context, args.batch_size), **measured},
            sort_keys=True,
        )
    )
    return 0


def _command(args: argparse.Namespace, operation: str, context: int, batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--operation",
        operation,
        "--context",
        str(context),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--attention-backend",
        args.attention_backend,
        "--d-model",
        str(args.d_model),
        "--num-heads",
        str(args.num_heads),
        "--rope-theta",
        str(args.rope_theta),
    ]


def _run_isolated(args: argparse.Namespace, operation: str, context: int, batch_size: int) -> dict[str, object]:
    child = subprocess.run(
        _command(args, operation, context, batch_size),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    lines = [line for line in child.stdout.splitlines() if line.strip()]
    if child.returncode != 0 or not lines:
        return {
            "status": "failed",
            "configuration": _configuration(args, operation, context, batch_size),
            "failure": {
                "type": "WorkerProcessError",
                "message": (child.stderr or child.stdout or f"worker exited {child.returncode}").strip(),
            },
        }
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as error:
        return {
            "status": "failed",
            "configuration": _configuration(args, operation, context, batch_size),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        return _worker(args)
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    if args.warmup < 0 or args.repetitions < 1:
        raise SystemExit("warmup must be non-negative and repetitions must be positive")
    allowed_operations = {"forward_only", "forward_backward"}
    if not args.operations or not set(args.operations) <= allowed_operations:
        raise SystemExit("operations must be forward_only and/or forward_backward")

    metadata = benchmark_metadata("attention", seed=args.seed, command=[sys.executable, *sys.argv])
    gpus = metadata["environment"]["gpus"]
    total_memory_bytes = gpus[0]["total_memory_bytes"] if gpus else None
    results = []
    for operation in args.operations:
        for context in args.contexts:
            for batch_size in args.batch_sizes:
                result = _run_isolated(args, operation, context, batch_size)
                result = enforce_device_memory_capacity(result, total_memory_bytes=total_memory_bytes)
                results.append(result)
                print(
                    f"operation={operation} context={context} batch={batch_size} status={result['status']}",
                    file=sys.stderr,
                    flush=True,
                )
                if result["status"] in {"oom", "memory_capacity_exceeded"}:
                    break

    report = {
        **metadata,
        "formal_benchmark": True,
        "isolation": "fresh subprocess, module, inputs, and RNG per configuration and operation",
        "configuration": {
            "contexts": args.contexts,
            "batch_sizes": args.batch_sizes,
            "operations": args.operations,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "precision": args.precision,
            "attention_backend": args.attention_backend,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "rope_theta": args.rope_theta,
        },
        "results": results,
    }
    write_report(
        report,
        json_path=args.output_dir / "benchmark.json",
        csv_path=args.output_dir / "benchmark.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
