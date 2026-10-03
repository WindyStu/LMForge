"""Optional torch.compile execution contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from lmforge.config import ModelConfig, TrainConfig


class _PrefixedStateWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


def _config(*, compile_model: bool) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(
            vocab_size=8,
            context_length=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            d_ff=16,
        ),
        max_steps=1,
        batch_size=1,
        grad_accum_steps=1,
        warmup_steps=0,
        save_interval=1,
        log_interval=1,
        compile_model=compile_model,
    )


def test_eager_is_default_and_compile_is_called_only_when_enabled(tmp_path, monkeypatch) -> None:
    from lmforge.training import engine
    from lmforge.training.logging import NullMetricsLogger

    calls = []

    def record_compile(model):
        calls.append(model)
        return _PrefixedStateWrapper(model)

    monkeypatch.setattr(torch, "compile", record_compile)
    data = np.arange(80) % 8
    eager = _config(compile_model=False)
    engine.train(
        eager,
        data,
        None,
        tmp_path / "eager",
        metrics_logger=NullMetricsLogger(),
    )
    assert calls == []

    compiled = engine.train(
        replace(eager, compile_model=True),
        data,
        None,
        tmp_path / "compiled",
        metrics_logger=NullMetricsLogger(),
    )

    assert len(calls) == 1
    assert compiled is not None
    assert not any(name.startswith("_orig_mod.") for name in compiled["model"])


def test_checkpoint_can_resume_with_compile_setting_changed(tmp_path, monkeypatch) -> None:
    from lmforge.training import engine
    from lmforge.training.logging import NullMetricsLogger

    monkeypatch.setattr(torch, "compile", _PrefixedStateWrapper)
    data = np.arange(80) % 8
    config = _config(compile_model=False)
    engine.train(
        config,
        data,
        None,
        tmp_path,
        metrics_logger=NullMetricsLogger(),
    )

    resumed = engine.train(
        replace(config, max_steps=2, compile_model=True),
        data,
        None,
        tmp_path,
        resume=tmp_path / "last.pt",
        metrics_logger=NullMetricsLogger(),
    )

    assert resumed is not None
    assert resumed["iteration"] == 2


def test_training_benchmark_configuration_records_compile_mode() -> None:
    from lmforge.benchmarking.workloads import training_configuration

    configuration = training_configuration(
        batch_size=1,
        context_length=128,
        warmup=2,
        repetitions=3,
        precision="bfloat16",
        attention_backend="sdpa",
        compile_model=True,
    )

    assert configuration["compile_model"] is True
    assert configuration["execution_mode"] == "compile"
