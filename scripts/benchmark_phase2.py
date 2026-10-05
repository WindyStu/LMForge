#!/usr/bin/env python3
"""Run the frozen LMForge Phase 2 publication benchmark protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lmforge.benchmarking.environment import benchmark_metadata
from lmforge.benchmarking.phase2 import formal_protocol, long_training_environment

ROOT = Path(__file__).resolve().parents[1]


def _variant_arguments(variant: dict[str, Any]) -> list[str]:
    arguments = [
        "--precision",
        str(variant["precision"]),
        "--attention-backend",
        str(variant["attention_backend"]),
    ]
    if variant["compile_model"]:
        arguments.append("--compile-model")
    return arguments


def _micro_command(
    script: str,
    output: Path,
    protocol: dict[str, Any],
    variant: dict[str, Any],
    *,
    capacity: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "--contexts",
        ",".join(map(str, protocol["contexts"])),
        "--batch-sizes",
        ",".join(map(str, protocol["batch_sizes"] if capacity else [1])),
        "--warmup",
        str(protocol["warmup"]),
        "--repetitions",
        str(protocol["steady_repetitions"]),
        "--seed",
        str(protocol["seed"]),
        "--device",
        "cuda",
        *_variant_arguments(variant),
        "--output-dir",
        str(output),
    ]
    if script == "benchmark_attention.py":
        command.extend(["--operations", "forward_only,forward_backward"])
    return command


def _commands(output: Path, protocol: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    runs = range(1, int(protocol["independent_runs"]) + 1)
    if stage in {"all", "control"}:
        for variant in protocol["control_variants"]:
            for run in runs:
                destination = output / "control" / variant["name"] / f"run-{run:02d}"
                commands.append(
                    {
                        "stage": "control",
                        "variant": variant["name"],
                        "run": run,
                        "output": str(destination),
                        "command": _micro_command(
                            "benchmark_training.py", destination, protocol, variant
                        ),
                    }
                )
    if stage in {"all", "capacity"}:
        for variant in protocol["capacity_variants"]:
            for run in runs:
                destination = output / "capacity" / variant["name"] / f"run-{run:02d}"
                commands.append(
                    {
                        "stage": "capacity",
                        "variant": variant["name"],
                        "run": run,
                        "output": str(destination),
                        "command": _micro_command(
                            "benchmark_training.py",
                            destination,
                            protocol,
                            variant,
                            capacity=True,
                        ),
                    }
                )
    if stage in {"all", "attention"}:
        for variant in protocol["attention_variants"]:
            for run in runs:
                destination = output / "attention" / variant["name"] / f"run-{run:02d}"
                commands.append(
                    {
                        "stage": "attention",
                        "variant": variant["name"],
                        "run": run,
                        "output": str(destination),
                        "command": _micro_command(
                            "benchmark_attention.py", destination, protocol, variant
                        ),
                    }
                )
    if stage in {"all", "long"}:
        configs = {
            "reference-fp32-eager": ROOT / "configs" / "tinystories" / "phase1.toml",
            "sdpa-bf16-compile": ROOT / "configs" / "tinystories" / "phase2-optimized.toml",
        }
        executable = shutil.which("lmforge") or "lmforge"
        for variant in protocol["long_training"]["variants"]:
            for run in runs:
                destination = output / "long" / variant["name"] / f"run-{run:02d}"
                commands.append(
                    {
                        "stage": "long",
                        "variant": variant["name"],
                        "run": run,
                        "output": str(destination),
                        "command": [
                            executable,
                            "train",
                            "--config",
                            str(configs[variant["name"]]),
                            "--max-steps",
                            str(protocol["long_training"]["steps"]),
                            "--output-dir",
                            str(destination),
                        ],
                    }
                )
    return commands


def _run(command: dict[str, Any]) -> dict[str, Any]:
    destination = Path(command["output"])
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    options = {
        "cwd": ROOT,
        "text": True,
        "encoding": "utf-8",
        "capture_output": True,
    }
    if command["stage"] == "long" and command["variant"].endswith("-compile"):
        with tempfile.TemporaryDirectory(prefix="lmforge-long-inductor-") as cache:
            environment = long_training_environment(
                os.environ,
                compile_model=True,
                cache_directory=cache,
            )
            completed = subprocess.run(
                command["command"], env=environment, check=False, **options
            )
    elif command["stage"] == "long":
        environment = long_training_environment(os.environ, compile_model=False)
        completed = subprocess.run(
            command["command"], env=environment, check=False, **options
        )
    else:
        completed = subprocess.run(command["command"], check=False, **options)
    (destination / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (destination / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    result = {
        **command,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "failed",
    }
    (destination / "execution.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "control", "capacity", "attention", "long"),
        default="all",
    )
    parser.add_argument("--independent-runs", type=int, default=3)
    parser.add_argument("--long-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = formal_protocol(
        independent_runs=args.independent_runs,
        long_training_steps=args.long_steps,
    )
    commands = _commands(args.output_dir.resolve(), protocol, args.stage)
    manifest = {
        **benchmark_metadata("phase2_publication_protocol", seed=protocol["seed"]),
        "protocol": protocol,
        "stage": args.stage,
        "commands": commands,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output_dir / f"protocol-{args.stage}.json"
    if protocol_path.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {protocol_path}")
    protocol_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results = []
    for index, command in enumerate(commands, 1):
        print(
            f"[{index}/{len(commands)}] {command['stage']} {command['variant']} run={command['run']}",
            flush=True,
        )
        try:
            results.append(_run(command))
        except Exception as error:  # noqa: BLE001 - preserve orchestration failure
            results.append(
                {
                    **command,
                    "status": "failed",
                    "failure": {"type": type(error).__name__, "message": str(error)},
                }
            )
    (args.output_dir / f"executions-{args.stage}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if all(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
