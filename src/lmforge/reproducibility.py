"""Randomness policy, environment capture, and run-manifest persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch


RUN_MANIFEST_FORMAT = "lmforge-run-manifest-v1"


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed every supported RNG and apply the requested PyTorch policy."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def _git_state(cwd: str | Path | None = None) -> dict[str, object]:
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=working_directory,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if commit.returncode != 0:
        return {"available": False, "commit": None, "dirty": None}
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=working_directory,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return {
        "available": status.returncode == 0,
        "commit": commit.stdout.strip(),
        "dirty": None if status.returncode != 0 else bool(status.stdout.strip()),
    }


def _cpu_name() -> str:
    name = platform.processor()
    if name:
        return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.machine() or "unknown"


def _environment() -> dict[str, object]:
    gpus = []
    if torch.cuda.is_available():
        gpus = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cpu": {
            "name": _cpu_name(),
            "logical_cores": os.cpu_count(),
        },
        "gpus": gpus,
    }


def build_run_manifest(
    *,
    seed: int,
    deterministic: bool,
    config: Mapping[str, object],
    datasets: Mapping[str, object],
    tokenizer_sha256: str | None,
    model_parameters: int,
    resume: str | Path | None,
    repository: str | Path | None = None,
) -> dict[str, object]:
    """Build a JSON-compatible snapshot of run identity and environment."""

    return {
        "format": RUN_MANIFEST_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "seeds": {
            "python": seed,
            "numpy_global": seed,
            "torch": seed,
            "cuda": seed,
            "train_numpy": seed,
            "validation_numpy": seed + 1,
        },
        "deterministic": deterministic,
        "config": dict(config),
        "git": _git_state(repository),
        "environment": _environment(),
        "datasets": dict(datasets),
        "tokenizer_sha256": tokenizer_sha256,
        "model_parameters": model_parameters,
        "invocation": list(sys.argv),
        "resume": None if resume is None else str(Path(resume)),
    }


def write_run_manifest(manifest: Mapping[str, object], path: str | Path) -> None:
    """Atomically write a run manifest as stable UTF-8 JSON."""

    destination = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".json.tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
