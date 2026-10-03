#!/usr/bin/env python3
"""Benchmark LMForge training execution strategies or profile the FP32 reference."""

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
    profile_training_workload,
    run_training_workload,
    training_configuration,
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
    parser.add_argument(
        "--batch-sizes",
        type=_positive_csv,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--attention-backend",
        choices=("reference", "naive", "sdpa"),
        default="reference",
    )
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", action="store_true", help="capture a profiler trace; never emits throughput")
    parser.add_argument("--profile-context", type=int, default=512)
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    return parser


def _workload_arguments(args: argparse.Namespace, context: int, batch_size: int) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "context_length": context,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "device": args.device,
        "precision": args.precision,
        "attention_backend": args.attention_backend,
        "seed": args.seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
    }


def _worker(args: argparse.Namespace) -> int:
    if args.context is None or args.batch_size is None:
        raise SystemExit("--worker requires --context and --batch-size")
    configuration = training_configuration(
        batch_size=args.batch_size,
        context_length=args.context,
        warmup=args.warmup,
        repetitions=args.repetitions,
        precision=args.precision,
        attention_backend=args.attention_backend,
    )
    measured = run_configuration(
        lambda: run_training_workload(**_workload_arguments(args, args.context, args.batch_size)),
        device=args.device,
    )
    print(json.dumps({"configuration": configuration, **measured}, sort_keys=True))
    return 0


def _worker_command(args: argparse.Namespace, context: int, batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
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
        "--vocab-size",
        str(args.vocab_size),
        "--d-model",
        str(args.d_model),
        "--num-layers",
        str(args.num_layers),
        "--num-heads",
        str(args.num_heads),
        "--d-ff",
        str(args.d_ff),
        "--rope-theta",
        str(args.rope_theta),
    ]


def _run_isolated(args: argparse.Namespace, context: int, batch_size: int) -> dict[str, object]:
    command = _worker_command(args, context, batch_size)
    child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    lines = [line for line in child.stdout.splitlines() if line.strip()]
    if child.returncode != 0 or not lines:
        return {
            "status": "failed",
            "configuration": training_configuration(
                batch_size=batch_size,
                context_length=context,
                warmup=args.warmup,
                repetitions=args.repetitions,
                precision=args.precision,
                attention_backend=args.attention_backend,
            ),
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
            "failure": {"type": type(error).__name__, "message": str(error)},
            "configuration": {"batch_size": batch_size, "context_length": context},
        }


def _profile(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    output = args.output_dir
    metadata = benchmark_metadata(
        "training_profiler",
        seed=args.seed,
        command=[sys.executable, *sys.argv],
    )
    profile_result = run_configuration(
        lambda: profile_training_workload(
            trace_path=output / "torch-profiler-trace.json",
            operator_path=output / "operator-summary.csv",
            batch_size=args.profile_batch_size,
            context_length=args.profile_context,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=args.device,
            seed=args.seed,
            warmup=args.warmup,
            repetitions=args.repetitions,
        ),
        device=args.device,
    )
    report = {
        **metadata,
        "formal_benchmark": False,
        "configuration": {
            "batch_size": args.profile_batch_size,
            "context_length": args.profile_context,
            "precision": "float32",
            "attention_backend": "reference",
        },
        "profile": profile_result,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile_result, indent=2))
    return 0 if profile_result["status"] == "ok" else 1


def _benchmark(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        raise SystemExit("--output-dir is required")
    metadata = benchmark_metadata("training", seed=args.seed, command=[sys.executable, *sys.argv])
    gpus = metadata["environment"]["gpus"]
    total_memory_bytes = gpus[0]["total_memory_bytes"] if gpus else None
    results = []
    for context in args.contexts:
        for batch_size in args.batch_sizes:
            result = _run_isolated(args, context, batch_size)
            result = enforce_device_memory_capacity(result, total_memory_bytes=total_memory_bytes)
            results.append(result)
            print(
                f"context={context} batch={batch_size} status={result['status']}",
                file=sys.stderr,
                flush=True,
            )
            if result["status"] in {"oom", "memory_capacity_exceeded"}:
                break
    report = {
        **metadata,
        "formal_benchmark": True,
        "isolation": "fresh subprocess, model, optimizer, and RNG per configuration",
        "configuration": {
            "contexts": args.contexts,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "precision": args.precision,
            "attention_backend": args.attention_backend,
            "model": {
                "vocab_size": args.vocab_size,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "rope_theta": args.rope_theta,
            },
        },
        "results": results,
    }
    write_report(
        report,
        json_path=args.output_dir / "benchmark.json",
        csv_path=args.output_dir / "benchmark.csv",
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        raise SystemExit("warmup must be non-negative and repetitions must be positive")
    if args.worker:
        return _worker(args)
    if args.profile:
        return _profile(args)
    return _benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
