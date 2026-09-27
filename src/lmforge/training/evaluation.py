"""Independent validation loop and shared language-model loss helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager

import numpy as np
import numpy.typing as npt
import torch

from ..config import TrainConfig
from .data import get_batch
from .loss import cross_entropy


def autocast_context(config: TrainConfig) -> AbstractContextManager:
    """Return the configured autocast context without changing precision policy."""

    dtype = torch.float16 if config.precision == "float16" else torch.bfloat16
    return torch.autocast(
        torch.device(config.device).type,
        dtype=dtype,
        enabled=config.precision != "float32",
    )


def language_model_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the existing mean token-level cross-entropy loss."""

    logits = model(inputs).float()
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: npt.NDArray,
    config: TrainConfig,
    rng: np.random.Generator,
) -> float:
    """Average validation loss over configured batches and restore model mode."""

    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(config.eval_batches):
            inputs, targets = get_batch(
                data,
                config.batch_size,
                config.model.context_length,
                config.device,
                rng=rng,
            )
            with autocast_context(config):
                loss = language_model_loss(model, inputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            total += loss.item()
        return total / config.eval_batches
    finally:
        model.train(was_training)
