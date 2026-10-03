"""BF16 training execution contracts."""

from __future__ import annotations

import pytest
import torch

from lmforge.config import ModelConfig, TrainConfig
from lmforge.nn.transformer import TransformerLM
from lmforge.training.evaluation import autocast_context, language_model_loss
from lmforge.training.loss import clip_gradients
from lmforge.training.optimizer import AdamW


def _cuda_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@pytest.mark.skipif(not _cuda_bf16_available(), reason="CUDA BF16 is required")
@pytest.mark.parametrize("attention_backend", ["reference", "sdpa"])
def test_bf16_short_training_stays_finite_and_keeps_fp32_state(
    attention_backend: str,
) -> None:
    torch.manual_seed(19)
    config = TrainConfig(
        model=ModelConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=4,
            d_ff=32,
            attention_backend=attention_backend,
        ),
        device="cuda",
        precision="bfloat16",
    )
    model = TransformerLM(**vars(config.model), device="cuda")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, 32, (2, 8), device="cuda")
    targets = torch.randint(0, 32, (2, 8), device="cuda")
    losses = []

    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config):
            loss = language_model_loss(model, inputs, targets)
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None
        )
        clip_gradients(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(
        not torch.is_tensor(value) or value.dtype == torch.float32
        for state in optimizer.state.values()
        for value in state.values()
    )


@pytest.mark.skipif(not _cuda_bf16_available(), reason="CUDA BF16 is required")
def test_training_benchmark_accepts_bfloat16() -> None:
    from lmforge.benchmarking.workloads import run_training_workload

    result = run_training_workload(
        batch_size=1,
        context_length=8,
        vocab_size=32,
        d_model=16,
        num_layers=1,
        num_heads=4,
        d_ff=32,
        rope_theta=10_000.0,
        device="cuda",
        precision="bfloat16",
        attention_backend="sdpa",
        seed=23,
        warmup=1,
        repetitions=2,
    )

    assert result["model_parameters"] > 0
    assert result["measurement"]["steady_state"]["repetitions"] == 2
