"""Training engine ownership and optimizer-step lifecycle contracts."""

import importlib.util
import inspect

import numpy as np
import torch

from lmforge.config import ModelConfig, TrainConfig


def test_training_engine_has_an_independent_module() -> None:
    assert importlib.util.find_spec("lmforge.training.engine") is not None


def test_legacy_train_module_is_only_a_compatibility_entrypoint() -> None:
    from lmforge.training import engine, train as train_module

    source = inspect.getsource(train_module)
    assert train_module.train is engine.train
    assert "for step in" not in source
    assert "TransformerLM" not in source


def test_explicit_rng_batch_matches_original_sampling_algorithm() -> None:
    from lmforge.training.data import get_batch

    data = np.arange(80) % 11
    seed = 23
    expected_rng = np.random.default_rng(seed)
    starts = expected_rng.integers(0, len(data) - 4, size=3)
    positions = starts[:, None] + np.arange(5)[None, :]
    expected = torch.tensor(data[positions], dtype=torch.long)

    inputs, targets = get_batch(
        data, 3, 4, "cpu", rng=np.random.default_rng(seed)
    )

    torch.testing.assert_close(inputs, expected[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(targets, expected[:, 1:], rtol=0, atol=0)


def test_optimizer_step_lifecycle_order_is_preserved(tmp_path, monkeypatch) -> None:
    from lmforge.training import engine
    from lmforge.training.logging import NullMetricsLogger

    events = []
    real_schedule = engine.cosine_learning_rate_schedule
    real_clip = engine.clip_gradients
    real_optimizer = engine.AdamW

    def record_schedule(*args, **kwargs):
        events.append("schedule")
        return real_schedule(*args, **kwargs)

    def record_clip(*args, **kwargs):
        events.append("clip")
        return real_clip(*args, **kwargs)

    class RecordingAdamW(real_optimizer):
        def zero_grad(self, *args, **kwargs):
            events.append("zero_grad")
            return super().zero_grad(*args, **kwargs)

        def step(self, *args, **kwargs):
            events.append("optimizer_step")
            return super().step(*args, **kwargs)

    monkeypatch.setattr(engine, "cosine_learning_rate_schedule", record_schedule)
    monkeypatch.setattr(engine, "clip_gradients", record_clip)
    monkeypatch.setattr(engine, "AdamW", RecordingAdamW)
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

    engine.train(
        config,
        np.arange(80) % 8,
        None,
        tmp_path,
        metrics_logger=NullMetricsLogger(),
    )

    assert events == ["schedule", "zero_grad", "clip", "optimizer_step"]
