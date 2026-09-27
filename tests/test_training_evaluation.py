"""Independent validation-loop contracts."""

import importlib.util

import numpy as np
import pytest
import torch

from lmforge.config import ModelConfig, TrainConfig


def test_evaluation_has_an_independent_module() -> None:
    assert importlib.util.find_spec("lmforge.training.evaluation") is not None


class _RecordingUniformLM(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))
        self.seen: list[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert not self.training
        self.seen.append(inputs.clone())
        return self.logits.expand(*inputs.shape, -1)


@pytest.mark.parametrize("starts_in_training_mode", [True, False])
def test_evaluation_does_not_update_parameters_and_restores_mode(
    starts_in_training_mode: bool,
) -> None:
    from lmforge.training.evaluation import evaluate

    config = TrainConfig(
        model=ModelConfig(vocab_size=4, context_length=3),
        batch_size=2,
        eval_batches=3,
    )
    model = _RecordingUniformLM(4).train(starts_in_training_mode)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    loss = evaluate(model, np.arange(64) % 4, config, np.random.default_rng(9))

    assert loss == pytest.approx(np.log(4))
    assert model.training is starts_in_training_mode
    assert len(model.seen) == config.eval_batches
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_training_flow_uses_independent_evaluation(tmp_path, monkeypatch) -> None:
    from lmforge.training import evaluation
    from lmforge.training.train import train

    calls = []

    def record_evaluation(model, data, config, rng):
        calls.append((model.training, len(data), config.eval_batches))
        return 1.5

    monkeypatch.setattr(evaluation, "evaluate", record_evaluation)
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
        eval_interval=1,
        eval_batches=2,
        save_interval=1,
        log_interval=1,
    )

    result = train(config, np.arange(80) % 8, np.arange(80) % 8, tmp_path)

    assert calls == [(True, 80, 2)]
    assert result["best_val_loss"] == 1.5
