"""Replaceable training metrics logging contracts."""

import importlib.util
from io import StringIO
import json

import numpy as np
import pytest

from lmforge.config import ModelConfig, TrainConfig


def test_training_logging_has_an_independent_module() -> None:
    assert importlib.util.find_spec("lmforge.training.logging") is not None


def test_jsonl_logger_preserves_fields_flushes_and_controls_stdout(tmp_path) -> None:
    from lmforge.training.logging import JsonlMetricsLogger

    path = tmp_path / "metrics.jsonl"
    stream = StringIO()
    logger = JsonlMetricsLogger(path, stream=stream)
    first = {"step": 1, "train_loss": 2.5, "lr": 0.01}
    second = {"step": 2, "train_loss": 2.0, "val_loss": 1.75}

    logger.log(first, emit_stdout=False)
    assert json.loads(path.read_text(encoding="utf-8")) == first
    assert stream.getvalue() == ""
    logger.log(second, emit_stdout=True)
    logger.close()

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [first, second]
    assert json.loads(stream.getvalue()) == second


def test_resume_metrics_rejects_steps_newer_than_checkpoint(tmp_path) -> None:
    from lmforge.training.logging import validate_resume_metrics

    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 3}\n', encoding="utf-8")

    validate_resume_metrics(path, 3)
    with pytest.raises(ValueError, match="newer than resume checkpoint"):
        validate_resume_metrics(path, 2)


class _RecordingLogger:
    def __init__(self) -> None:
        self.records = []
        self.closed = False

    def log(self, record, *, emit_stdout: bool) -> None:
        self.records.append((dict(record), emit_stdout))

    def close(self) -> None:
        self.closed = True


def test_training_metrics_logger_can_be_replaced_or_disabled(tmp_path) -> None:
    from lmforge.training.logging import NullMetricsLogger
    from lmforge.training.train import train

    config = TrainConfig(
        model=ModelConfig(
            vocab_size=8,
            context_length=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            d_ff=16,
        ),
        max_steps=2,
        batch_size=1,
        grad_accum_steps=1,
        warmup_steps=0,
        save_interval=2,
        log_interval=2,
    )
    data = np.arange(80) % 8
    recording = _RecordingLogger()

    train(config, data, None, tmp_path / "recorded", metrics_logger=recording)
    train(
        config,
        data,
        None,
        tmp_path / "disabled",
        metrics_logger=NullMetricsLogger(),
    )

    assert [record["step"] for record, _ in recording.records] == [1, 2]
    assert [emit for _, emit in recording.records] == [False, True]
    assert all("train_loss" in record and "lr" in record for record, _ in recording.records)
    assert not recording.closed
    assert not (tmp_path / "recorded" / "metrics.jsonl").exists()
    assert not (tmp_path / "disabled" / "metrics.jsonl").exists()
