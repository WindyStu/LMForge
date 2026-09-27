"""Reproducibility policy and run-manifest contracts."""

import importlib.util
import json
from dataclasses import replace

import numpy as np
import torch

from lmforge.config import ModelConfig, TrainConfig
from lmforge.training import checkpoint
from lmforge.training.logging import NullMetricsLogger


def test_reproducibility_has_an_independent_module() -> None:
    assert importlib.util.find_spec("lmforge.reproducibility") is not None


def _config(*, seed: int = 31, max_steps: int = 2) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(
            vocab_size=8,
            context_length=4,
            d_model=8,
            num_layers=1,
            num_heads=2,
            d_ff=16,
        ),
        max_steps=max_steps,
        batch_size=2,
        grad_accum_steps=2,
        warmup_steps=0,
        lr_decay_steps=10,
        eval_interval=2,
        eval_batches=2,
        save_interval=2,
        log_interval=2,
        seed=seed,
        deterministic=True,
    )


def _assert_nested_equal(left, right) -> None:
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_seed_policy_resets_python_numpy_torch_and_deterministic_state() -> None:
    import random

    from lmforge.reproducibility import configure_reproducibility

    configure_reproducibility(19, True)
    first = (random.random(), np.random.random(), torch.rand(3))
    configure_reproducibility(19, True)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()

    configure_reproducibility(19, False)
    assert not torch.are_deterministic_algorithms_enabled()


def test_seed_policy_explicitly_seeds_all_cuda_devices(monkeypatch) -> None:
    from lmforge import reproducibility

    calls = []
    monkeypatch.setattr(reproducibility.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        reproducibility.torch.cuda,
        "manual_seed_all",
        lambda seed: calls.append(seed),
    )

    reproducibility.configure_reproducibility(23, False)

    assert 23 in calls


def test_engine_uses_seed_policy_and_writes_complete_manifest(tmp_path, monkeypatch) -> None:
    from lmforge import reproducibility
    from lmforge.tokenization.tokenizer import BPE_tokenizer
    from lmforge.training import engine

    calls = []
    real_configure = reproducibility.configure_reproducibility

    def record_configure(seed: int, deterministic: bool) -> None:
        calls.append((seed, deterministic))
        real_configure(seed, deterministic)

    monkeypatch.setattr(reproducibility, "configure_reproducibility", record_configure)
    config = _config(max_steps=1)
    project_config = {"complete": {"runtime": {"deterministic": True}}}
    data = np.arange(80) % 8
    tokenizer = BPE_tokenizer({index: bytes([index]) for index in range(8)}, [])

    engine.train(
        config,
        data,
        data,
        tmp_path,
        tokenizer=tokenizer,
        metrics_logger=NullMetricsLogger(),
        manifest_config=project_config,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert calls == [(config.seed, True)]
    assert manifest["format"] == "lmforge-run-manifest-v1"
    assert manifest["seed"] == config.seed
    assert manifest["deterministic"] is True
    assert manifest["config"] == project_config
    assert set(manifest["git"]) == {"available", "commit", "dirty"}
    assert {
        "python",
        "torch",
        "cuda",
        "cpu",
        "gpus",
    } <= manifest["environment"].keys()
    assert len(manifest["datasets"]["train"]["sha256"]) == 64
    assert manifest["datasets"]["train"]["tokens"] == len(data)
    assert manifest["datasets"]["validation"]["tokens"] == len(data)
    assert len(manifest["tokenizer_sha256"]) == 64
    assert manifest["model_parameters"] > 0


def test_same_seed_cpu_short_training_is_exactly_reproducible(tmp_path) -> None:
    from lmforge.training.engine import train

    config = _config()
    data = np.arange(160) % 8

    first = train(
        config,
        data,
        data,
        tmp_path / "first",
        metrics_logger=NullMetricsLogger(),
    )
    second = train(
        config,
        data,
        data,
        tmp_path / "second",
        metrics_logger=NullMetricsLogger(),
    )

    for key in (
        "iteration",
        "model",
        "optimizer",
        "scaler",
        "best_val_loss",
        "data_sha256",
        "train_rng",
        "val_rng",
        "torch_rng",
        "python_rng",
        "cuda_rng",
    ):
        _assert_nested_equal(first[key], second[key])


def test_resume_accepts_v1_config_without_deterministic_field(tmp_path) -> None:
    from lmforge.training.engine import train

    config = _config(max_steps=1)
    data = np.arange(80) % 8
    state = train(
        config,
        data,
        None,
        tmp_path,
        metrics_logger=NullMetricsLogger(),
    )
    old_v1 = dict(state)
    old_v1["config"] = dict(state["config"])
    old_v1["config"].pop("deterministic")
    resume_path = tmp_path / "old-v1.pt"
    checkpoint.save_training_checkpoint(old_v1, resume_path)

    resumed = train(
        replace(config, max_steps=2, deterministic=False),
        data,
        None,
        tmp_path,
        resume=resume_path,
        metrics_logger=NullMetricsLogger(),
    )

    assert resumed["iteration"] == 2
