"""Benchmark provenance and runtime environment capture."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def git_state(repository: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd() if repository is None else Path(repository)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if revision.returncode != 0:
        return {"available": False, "sha": None, "dirty": None}
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return {
        "available": status.returncode == 0,
        "sha": revision.stdout.strip(),
        "dirty": None if status.returncode != 0 else bool(status.stdout.strip()),
    }


def runtime_environment() -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
    }


def benchmark_metadata(
    benchmark: str,
    *,
    seed: int,
    repository: str | Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "format": "lmforge-benchmark-v1",
        "benchmark": benchmark,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": list(sys.argv if command is None else command),
        "seed": seed,
        "git": git_state(repository),
        "environment": runtime_environment(),
    }
