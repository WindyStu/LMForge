"""Training state initialization and optimizer-step lifecycle."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from .. import reproducibility
from ..config import TrainConfig
from ..nn.transformer import TransformerLM
from ..tokenization.tokenizer import BPE_tokenizer
from . import checkpoint as checkpoint_io
from . import evaluation as evaluation_loop
from . import logging as metrics_logging
from .data import get_batch
from .loss import clip_gradients
from .optimizer import AdamW
from .prepare import tokenizer_fingerprint, tokenizer_state
from .schedule import cosine_learning_rate_schedule


def _validate_runtime_environment(config: TrainConfig) -> None:
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but this Python environment has no available CUDA device")
    if device.type == "cuda" and config.precision == "bfloat16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("this CUDA device does not support bfloat16")


def _validate_data(data: npt.NDArray, config: TrainConfig) -> str:
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("token data must be a 1D integer array")
    if len(data) <= config.model.context_length:
        raise ValueError("token data must contain at least context_length + 1 tokens")
    digest = hashlib.sha256()
    for start in range(0, len(data), 1_000_000):
        chunk = np.asarray(data[start : start + 1_000_000], dtype=np.int64)
        if chunk.min() < 0 or chunk.max() >= config.model.vocab_size:
            raise ValueError("token IDs outside model vocabulary")
        digest.update(chunk.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _checkpoint_compatibility_config(values: Mapping[str, object]) -> dict[str, object]:
    """Remove execution policies that do not change checkpoint tensor shapes."""

    compatible = copy.deepcopy(dict(values))
    for key in (
        "max_steps",
        "device",
        "precision",
        "compile_model",
        "log_interval",
        "save_interval",
    ):
        compatible.pop(key, None)
    model = compatible.get("model")
    if isinstance(model, dict):
        model.pop("attention_backend", None)
    compatible.setdefault("deterministic", False)
    return compatible


def train(
    config: TrainConfig,
    train_tokens: npt.NDArray,
    val_tokens: npt.NDArray | None,
    output_dir: str | Path,
    *,
    tokenizer: BPE_tokenizer | None = None,
    resume: str | Path | None = None,
    metrics_logger: metrics_logging.MetricsLogger | None = None,
    manifest_config: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Train to ``max_steps`` total and return the last checkpoint state."""

    config.validate()
    _validate_runtime_environment(config)
    data_hashes = {
        "train": _validate_data(train_tokens, config),
        "validation": (_validate_data(val_tokens, config) if val_tokens is not None else None),
    }
    if tokenizer is not None and set(tokenizer.vocab) != set(range(config.model.vocab_size)):
        raise ValueError("tokenizer IDs must exactly cover the configured model vocabulary")
    output = Path(output_dir)
    if resume is None and any(
        (output / name).exists() for name in ("last.pt", "best.pt", "metrics.jsonl", "manifest.json")
    ):
        raise FileExistsError("output contains a run; use --resume or a new output directory")
    state = None
    if resume is not None:
        state = checkpoint_io.load_training_checkpoint(resume)
        old_config = _checkpoint_compatibility_config(state["config"])
        new_config = _checkpoint_compatibility_config(asdict(config))
        if old_config != new_config or state["data_sha256"] != data_hashes:
            raise ValueError("resume configuration or training/validation data differ from checkpoint")
        if config.max_steps <= state["iteration"]:
            raise ValueError("max_steps must exceed the checkpoint iteration")
        saved_tok = state.get("tokenizer")
        if tokenizer is None and saved_tok is not None:
            tokenizer = BPE_tokenizer(**saved_tok)
        elif tokenizer is not None and tokenizer_state(tokenizer) != saved_tok:
            raise ValueError("resume tokenizer differs from checkpoint")
        metrics_logging.validate_resume_metrics(output / "metrics.jsonl", state["iteration"])

    reproducibility.configure_reproducibility(config.seed, config.deterministic)
    train_rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 1)
    model = TransformerLM(**asdict(config.model), device=config.device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "float16")
    iteration, best_val_loss = 0, float("inf")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        iteration, best_val_loss = state["iteration"], state["best_val_loss"]
        train_rng.bit_generator.state = state["train_rng"]
        val_rng.bit_generator.state = state["val_rng"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if torch.device(config.device).type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state

    model.train()
    execution_model = torch.compile(model) if config.compile_model else model

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    datasets = {
        "train": {
            "sha256": data_hashes["train"],
            "tokens": len(train_tokens),
            "dtype": str(train_tokens.dtype),
        },
        "validation": (
            {
                "sha256": data_hashes["validation"],
                "tokens": len(val_tokens),
                "dtype": str(val_tokens.dtype),
            }
            if val_tokens is not None
            else None
        ),
    }
    manifest = reproducibility.build_run_manifest(
        seed=config.seed,
        deterministic=config.deterministic,
        config=manifest_config if manifest_config is not None else asdict(config),
        datasets=datasets,
        tokenizer_sha256=(tokenizer_fingerprint(tokenizer) if tokenizer is not None else None),
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        resume=resume,
    )
    reproducibility.write_run_manifest(manifest, output / "manifest.json")
    device = torch.device(config.device)
    checkpoint = None
    with metrics_logging.metrics_logger_context(output / "metrics.jsonl", metrics_logger) as metrics:
        for step in range(iteration + 1, config.max_steps + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            # Schedule index is the zero-based optimizer attempt, matching the lab helper.
            lr = cosine_learning_rate_schedule(
                step - 1,
                config.max_lr,
                config.min_lr,
                config.warmup_steps,
                config.lr_decay_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(config.grad_accum_steps):
                inputs, targets = get_batch(
                    train_tokens,
                    config.batch_size,
                    config.model.context_length,
                    config.device,
                    rng=train_rng,
                )
                with evaluation_loop.autocast_context(config):
                    loss = evaluation_loop.language_model_loss(
                        execution_model,
                        inputs,
                        targets,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                train_loss += loss.item() / config.grad_accum_steps
                scaler.scale(loss / config.grad_accum_steps).backward()
            scaler.unscale_(optimizer)
            grads_finite = all(
                torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            if grads_finite:
                clip_gradients(model.parameters(), config.max_grad_norm)
            elif not scaler.is_enabled():
                raise FloatingPointError(f"non-finite gradients at step {step}")
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            record = {
                "step": step,
                "train_loss": train_loss,
                "lr": lr,
                "optimizer_step_skipped": skipped,
                "train_seconds": seconds,
                "tokens_per_second": (
                    config.batch_size * config.grad_accum_steps * config.model.context_length / seconds
                ),
            }
            improved = False
            # Eval cadence is independent of max_steps so stopping/resuming does not
            # consume extra validation RNG draws at an intermediate final checkpoint.
            if val_tokens is not None and step % config.eval_interval == 0:
                val_loss = evaluation_loop.evaluate(
                    execution_model,
                    val_tokens,
                    config,
                    val_rng,
                )
                record["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss, improved = val_loss, True
            metrics.log(
                record,
                emit_stdout=(step % config.log_interval == 0 or step == config.max_steps),
            )
            if step % config.save_interval == 0 or step == config.max_steps or improved:
                checkpoint = checkpoint_io.create_training_checkpoint(
                    iteration=step,
                    config=asdict(config),
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                    best_val_loss=best_val_loss,
                    data_sha256=data_hashes,
                    tokenizer=(tokenizer_state(tokenizer) if tokenizer is not None else None),
                    train_rng=train_rng.bit_generator.state,
                    val_rng=val_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    python_rng=random.getstate(),
                    cuda_rng=(torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
                )
                checkpoint_io.save_training_checkpoint(checkpoint, output / "last.pt")
                if improved:
                    checkpoint_io.save_training_checkpoint(checkpoint, output / "best.pt")
    return checkpoint
