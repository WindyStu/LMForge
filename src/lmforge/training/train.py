"""Train the assignment Transformer LM on memory-mapped token arrays."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..config import ModelConfig, TrainConfig
from ..nn.transformer import TransformerLM
from ..tokenization.tokenizer import BPE_tokenizer
from . import checkpoint as checkpoint_io
from .loss import clip_gradients, cross_entropy
from .optimizer import AdamW
from .prepare import tokenizer_state
from .schedule import cosine_learning_rate_schedule

def _validate_runtime_environment(config):
    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested, but this Python environment has no available CUDA device')
    if device.type == 'cuda' and config.precision == 'bfloat16':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('this CUDA device does not support bfloat16')


def _validate_data(data, config):
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError('token data must be a 1D integer array')
    if len(data) <= config.model.context_length:
        raise ValueError('token data must contain at least context_length + 1 tokens')
    digest = hashlib.sha256()
    for start in range(0, len(data), 1_000_000):
        chunk = np.asarray(data[start:start + 1_000_000], dtype=np.int64)
        if chunk.min() < 0 or chunk.max() >= config.model.vocab_size:
            raise ValueError('token IDs outside model vocabulary')
        digest.update(chunk.astype('<i8', copy=False).tobytes())
    return digest.hexdigest()


def _batch(data, config, rng):
    length = config.model.context_length
    starts = rng.integers(0, len(data) - length, size=config.batch_size)
    positions = starts[:, None] + np.arange(length + 1)[None, :]
    tokens = torch.tensor(data[positions], dtype=torch.long, device=config.device)
    return tokens[:, :-1], tokens[:, 1:]


def _autocast(config):
    dtype = torch.float16 if config.precision == 'float16' else torch.bfloat16
    return torch.autocast(torch.device(config.device).type, dtype=dtype,
                          enabled=config.precision != 'float32')


def _loss(model, inputs, targets):
    logits = model(inputs).float()
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(model, data, config, rng):
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(config.eval_batches):
            inputs, targets = _batch(data, config, rng)
            with _autocast(config):
                loss = _loss(model, inputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite validation loss')
            total += loss.item()
        return total / config.eval_batches
    finally:
        model.train(was_training)


def train(config, train_tokens, val_tokens, output_dir, *, tokenizer=None, resume=None):
    """Train to max_steps (total, not additional steps), saving last.pt / best.pt.

    A step comprises grad_accum_steps equally sized microbatches. FP16 uses a
    GradScaler; overflow attempts are logged as skipped optimizer steps. CPU
    float32 resume restores RNG and optimizer states for reproducible continuation.
    """
    config.validate()
    _validate_runtime_environment(config)
    data_hashes = {'train': _validate_data(train_tokens, config),
                   'validation': _validate_data(val_tokens, config) if val_tokens is not None else None}
    if tokenizer is not None and set(tokenizer.vocab) != set(range(config.model.vocab_size)):
        raise ValueError('tokenizer IDs must exactly cover the configured model vocabulary')
    output = Path(output_dir)
    if resume is None and any((output / name).exists() for name in ('last.pt', 'best.pt', 'metrics.jsonl')):
        raise FileExistsError('output contains a run; use --resume or a new output directory')
    state = None
    if resume is not None:
        state = checkpoint_io.load_training_checkpoint(resume)
        old_config, new_config = dict(state['config']), asdict(config)
        for key in ('max_steps', 'device', 'log_interval', 'save_interval'):
            old_config.pop(key)
            new_config.pop(key)
        if old_config != new_config or state['data_sha256'] != data_hashes:
            raise ValueError('resume configuration or training/validation data differ from checkpoint')
        if config.max_steps <= state['iteration']:
            raise ValueError('max_steps must exceed the checkpoint iteration')
        saved_tok = state.get('tokenizer')
        if tokenizer is None and saved_tok is not None:
            tokenizer = BPE_tokenizer(**saved_tok)
        elif tokenizer is not None and tokenizer_state(tokenizer) != saved_tok:
            raise ValueError('resume tokenizer differs from checkpoint')
        log_path = output / 'metrics.jsonl'
        if log_path.exists():
            with log_path.open(encoding='utf-8') as log:
                if any(json.loads(line)['step'] > state['iteration'] for line in log if line.strip()):
                    raise ValueError('output log is newer than resume checkpoint; use a new output directory')

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 1)
    model = TransformerLM(**asdict(config.model), device=config.device)
    optimizer = AdamW(model.parameters(), lr=config.max_lr,
                      betas=(config.beta1, config.beta2), eps=config.eps,
                      weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.precision == 'float16')
    iteration, best_val_loss = 0, float('inf')
    if state is not None:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        iteration, best_val_loss = state['iteration'], state['best_val_loss']
        train_rng.bit_generator.state = state['train_rng']
        val_rng.bit_generator.state = state['val_rng']
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if torch.device(config.device).type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state

    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    model.train()
    device = torch.device(config.device)
    checkpoint = None
    with (output / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for step in range(iteration + 1, config.max_steps + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            # Schedule index is the zero-based optimizer attempt, matching the lab helper.
            lr = cosine_learning_rate_schedule(step - 1, config.max_lr, config.min_lr,
                                               config.warmup_steps, config.lr_decay_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(config.grad_accum_steps):
                inputs, targets = _batch(train_tokens, config, train_rng)
                with _autocast(config):
                    loss = _loss(model, inputs, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite training loss at step {step}')
                train_loss += loss.item() / config.grad_accum_steps
                scaler.scale(loss / config.grad_accum_steps).backward()
            scaler.unscale_(optimizer)
            grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            if grads_finite:
                clip_gradients(model.parameters(), config.max_grad_norm)
            elif not scaler.is_enabled():
                raise FloatingPointError(f'non-finite gradients at step {step}')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            record = {'step': step, 'train_loss': train_loss, 'lr': lr,
                      'optimizer_step_skipped': skipped, 'train_seconds': seconds,
                      'tokens_per_second': config.batch_size * config.grad_accum_steps * config.model.context_length / seconds}
            improved = False
            # Eval cadence is independent of max_steps so stopping/resuming does not
            # consume extra validation RNG draws at an intermediate final checkpoint.
            if val_tokens is not None and step % config.eval_interval == 0:
                val_loss = evaluate(model, val_tokens, config, val_rng)
                record['val_loss'] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss, improved = val_loss, True
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            if step % config.log_interval == 0 or step == config.max_steps:
                print(json.dumps(record), flush=True)
            if step % config.save_interval == 0 or step == config.max_steps or improved:
                checkpoint = checkpoint_io.create_training_checkpoint(
                    iteration=step,
                    config=asdict(config),
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                    best_val_loss=best_val_loss,
                    data_sha256=data_hashes,
                    tokenizer=tokenizer_state(tokenizer) if tokenizer is not None else None,
                    train_rng=train_rng.bit_generator.state,
                    val_rng=val_rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                )
                checkpoint_io.save_training_checkpoint(checkpoint, output / 'last.pt')
                if improved:
                    checkpoint_io.save_training_checkpoint(checkpoint, output / 'best.pt')
    return checkpoint


def main():
    from ..cli import legacy_main

    return legacy_main("train")


if __name__ == '__main__':
    raise SystemExit(main())
