"""Replaceable metrics sinks with the existing JSONL and stdout semantics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Mapping, Protocol, TextIO


class MetricsLogger(Protocol):
    """Destination for one already-assembled training metrics record."""

    def log(self, record: Mapping[str, object], *, emit_stdout: bool) -> None: ...

    def close(self) -> None: ...


class JsonlMetricsLogger:
    """Append metrics as flushed JSONL and optionally mirror records to stdout."""

    def __init__(self, path: str | Path, *, stream: TextIO | None = None) -> None:
        self._log = Path(path).open("a", encoding="utf-8")
        self._stream = stream

    def log(self, record: Mapping[str, object], *, emit_stdout: bool) -> None:
        serialized = json.dumps(dict(record), allow_nan=False)
        self._log.write(serialized + "\n")
        self._log.flush()
        if emit_stdout:
            print(serialized, file=self._stream or sys.stdout, flush=True)

    def close(self) -> None:
        self._log.close()


class NullMetricsLogger:
    """Metrics sink that deliberately discards every record."""

    def log(self, record: Mapping[str, object], *, emit_stdout: bool) -> None:
        return None

    def close(self) -> None:
        return None


@contextmanager
def metrics_logger_context(
    path: str | Path,
    logger: MetricsLogger | None,
) -> Iterator[MetricsLogger]:
    """Yield an injected logger or own the lifecycle of the default JSONL logger."""

    if logger is not None:
        yield logger
        return
    owned = JsonlMetricsLogger(path)
    try:
        yield owned
    finally:
        owned.close()


def validate_resume_metrics(path: str | Path, iteration: int) -> None:
    """Reject appending when an existing metrics log is newer than a checkpoint."""

    log_path = Path(path)
    if not log_path.exists():
        return
    with log_path.open(encoding="utf-8") as log:
        if any(
            json.loads(line)["step"] > iteration
            for line in log
            if line.strip()
        ):
            raise ValueError(
                "output log is newer than resume checkpoint; use a new output directory"
            )
