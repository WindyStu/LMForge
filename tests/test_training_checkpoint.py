"""Training checkpoint compatibility and round-trip contracts."""

from io import BytesIO
from dataclasses import replace

import numpy as np
import torch

from lmforge.config import ModelConfig, TrainConfig
from lmforge.training import checkpoint
from lmforge.training.optimizer import AdamW


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


def test_training_checkpoint_api_is_owned_by_checkpoint_module() -> None:
    assert hasattr(checkpoint, "load_training_checkpoint")
    assert hasattr(checkpoint, "save_training_checkpoint")


def _model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(3, 2)
    optimizer = AdamW(model.parameters())
    inputs = torch.ones(1, 3)
    model(inputs).sum().backward()
    optimizer.step()
    return model, optimizer


def test_load_checkpoint_supports_legacy_unversioned_files() -> None:
    model, optimizer = _model_and_optimizer()
    legacy = BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": 7,
        },
        legacy,
    )
    legacy.seek(0)

    restored_model, restored_optimizer = _model_and_optimizer()
    iteration = checkpoint.load_checkpoint(legacy, restored_model, restored_optimizer)

    assert iteration == 7
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored_model.state_dict()[name])
    assert optimizer.state_dict()["param_groups"] == restored_optimizer.state_dict()["param_groups"]


def test_complete_training_checkpoint_round_trip(tmp_path) -> None:
    model, optimizer = _model_and_optimizer()
    state = {
        "format": checkpoint.TRAINING_CHECKPOINT_FORMAT,
        "iteration": 3,
        "config": {"model": {}},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": {},
        "best_val_loss": 1.25,
        "data_sha256": {"train": "abc", "validation": None},
        "tokenizer": None,
        "train_rng": np.random.default_rng(1).bit_generator.state,
        "val_rng": np.random.default_rng(2).bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "python_rng": (3, (), None),
        "cuda_rng": None,
    }
    path = tmp_path / "last.pt"

    checkpoint.save_training_checkpoint(state, path)
    restored = checkpoint.load_training_checkpoint(path)

    assert restored.keys() == state.keys()
    assert restored["iteration"] == 3
    assert restored["data_sha256"] == state["data_sha256"]
    for name, value in state["model"].items():
        torch.testing.assert_close(value, restored["model"][name])


def test_training_flow_saves_through_checkpoint_module(tmp_path, monkeypatch) -> None:
    from lmforge.training.train import train

    calls = []
    real_save = checkpoint.save_training_checkpoint

    def record_save(state, out):
        calls.append(out)
        real_save(state, out)

    monkeypatch.setattr(checkpoint, "save_training_checkpoint", record_save)
    config = TrainConfig(
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
    )

    train(config, np.arange(80) % 8, None, tmp_path)

    assert calls == [tmp_path / "last.pt"]


def test_training_resume_loads_through_checkpoint_module(tmp_path, monkeypatch) -> None:
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
        max_steps=1,
        batch_size=1,
        grad_accum_steps=1,
        warmup_steps=0,
        save_interval=1,
        log_interval=1,
    )
    data = np.arange(80) % 8
    train(config, data, None, tmp_path)
    resume_path = tmp_path / "last.pt"
    calls = []
    real_load = checkpoint.load_training_checkpoint

    def record_load(src):
        calls.append(src)
        return real_load(src)

    monkeypatch.setattr(checkpoint, "load_training_checkpoint", record_load)

    train(replace(config, max_steps=2), data, None, tmp_path, resume=resume_path)

    assert calls == [resume_path]


def test_generation_loads_through_checkpoint_module(tmp_path, monkeypatch) -> None:
    from lmforge.training.generate import load_model
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
        max_steps=1,
        batch_size=1,
        grad_accum_steps=1,
        warmup_steps=0,
        save_interval=1,
        log_interval=1,
    )
    train(config, np.arange(80) % 8, None, tmp_path)
    path = tmp_path / "last.pt"
    calls = []
    real_load = checkpoint.load_training_checkpoint

    def record_load(src):
        calls.append(src)
        return real_load(src)

    monkeypatch.setattr(checkpoint, "load_training_checkpoint", record_load)

    load_model(path)

    assert calls == [path]


def test_interrupted_training_matches_uninterrupted_checkpoint_state(tmp_path) -> None:
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
        max_steps=4,
        batch_size=2,
        grad_accum_steps=2,
        warmup_steps=0,
        lr_decay_steps=10,
        eval_interval=2,
        eval_batches=2,
        save_interval=2,
        log_interval=4,
        seed=17,
    )
    data = np.arange(160) % 8

    uninterrupted = train(config, data, data, tmp_path / "uninterrupted")
    interrupted_dir = tmp_path / "interrupted"
    train(replace(config, max_steps=2), data, data, interrupted_dir)
    resumed = train(
        config,
        data,
        data,
        interrupted_dir,
        resume=interrupted_dir / "last.pt",
    )

    assert uninterrupted is not None
    assert resumed is not None
    for key in (
        "iteration",
        "model",
        "optimizer",
        "scaler",
        "best_val_loss",
        "data_sha256",
        "tokenizer",
        "train_rng",
        "val_rng",
        "torch_rng",
        "python_rng",
        "cuda_rng",
    ):
        _assert_nested_equal(uninterrupted[key], resumed[key])
