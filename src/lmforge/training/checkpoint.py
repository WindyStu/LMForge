import os
from pathlib import Path
import tempfile
from typing import IO, BinaryIO, Mapping

import torch


TRAINING_CHECKPOINT_FORMAT = "cs336-training-v1"
TRAINING_CHECKPOINT_KEYS = frozenset(
    {
        "format",
        "iteration",
        "config",
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
    }
)


def _validate_training_checkpoint(state: object) -> dict[str, object]:
    if not isinstance(state, dict) or state.get("format") != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError(f"expected a {TRAINING_CHECKPOINT_FORMAT} checkpoint")
    missing = TRAINING_CHECKPOINT_KEYS.difference(state)
    if missing:
        raise ValueError(f"training checkpoint is missing field: {min(missing)}")
    return state


def _load_serialized(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, object]:
    state = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return state


def _save_serialized(
    state: Mapping[str, object],
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    if not isinstance(out, (str, os.PathLike)):
        torch.save(state, out)
        return

    path = Path(out)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".pt.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, object]:
    """Load and validate a complete checkpoint written by the training engine."""

    state = _load_serialized(src)
    return _validate_training_checkpoint(state)


def save_training_checkpoint(
    state: dict[str, object],
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Persist a complete training checkpoint, atomically when ``out`` is a path."""

    _save_serialized(_validate_training_checkpoint(state), out)


def create_training_checkpoint(
    *,
    iteration: int,
    config: Mapping[str, object],
    model: Mapping[str, object],
    optimizer: Mapping[str, object],
    scaler: Mapping[str, object],
    best_val_loss: float,
    data_sha256: Mapping[str, object],
    tokenizer: Mapping[str, object] | None,
    train_rng: Mapping[str, object],
    val_rng: Mapping[str, object],
    torch_rng: torch.Tensor,
    python_rng: tuple[object, ...],
    cuda_rng: list[torch.Tensor] | None,
) -> dict[str, object]:
    """Build the canonical complete training checkpoint schema."""

    return _validate_training_checkpoint(
        {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "iteration": iteration,
            "config": dict(config),
            "model": dict(model),
            "optimizer": dict(optimizer),
            "scaler": dict(scaler),
            "best_val_loss": best_val_loss,
            "data_sha256": dict(data_sha256),
            "tokenizer": None if tokenizer is None else dict(tokenizer),
            "train_rng": dict(train_rng),
            "val_rng": dict(val_rng),
            "torch_rng": torch_rng,
            "python_rng": python_rng,
            "cuda_rng": cuda_rng,
        }
    )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    _save_serialized(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    state_dict_load = _load_serialized(src)
    checkpoint_format = state_dict_load.get("format")
    if checkpoint_format is not None:
        _validate_training_checkpoint(state_dict_load)
    model.load_state_dict(state_dict_load["model"])
    optimizer.load_state_dict(state_dict_load["optimizer"])

    return state_dict_load["iteration"]
