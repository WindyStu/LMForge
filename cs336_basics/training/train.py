"""Train the assignment Transformer LM on memory-mapped token arrays."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.tokenization.tokenizer import BPE_tokenizer
from cs336_basics.training.loss import clip_gradients, cross_entropy
from cs336_basics.training.optimizer import AdamW
from cs336_basics.training.prepare import tokenizer_fingerprint, tokenizer_state
from cs336_basics.training.schedule import cosine_learning_rate_schedule


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1000
    context_length: int = 128
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 352
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    max_steps: int = 1000
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 50
    lr_decay_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 10
    save_interval: int = 100
    log_interval: int = 10
    seed: int = 42
    device: str = 'cpu'
    precision: str = 'float32'

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values['model'] = ModelConfig(**values.get('model', {}))
        return cls(**values)

    def validate(self):
        m = self.model
        integers = [m.vocab_size, m.context_length, m.d_model, m.num_layers, m.num_heads,
                    m.d_ff, self.max_steps, self.batch_size, self.grad_accum_steps,
                    self.lr_decay_steps, self.eval_interval, self.eval_batches,
                    self.save_interval, self.log_interval]
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError('model dimensions, batch sizes, steps and intervals must be positive integers')
        if m.d_model % m.num_heads or (m.d_model // m.num_heads) % 2:
            raise ValueError('d_model must be divisible by num_heads with an even RoPE head dimension')
        if not math.isfinite(m.rope_theta) or m.rope_theta <= 0:
            raise ValueError('rope_theta must be positive and finite')
        if not 0 <= self.warmup_steps < self.lr_decay_steps:
            raise ValueError('require 0 <= warmup_steps < lr_decay_steps')
        numeric = [self.max_lr, self.min_lr, self.weight_decay, self.eps, self.max_grad_norm]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('training hyperparameters must be finite')
        if not 0 <= self.min_lr <= self.max_lr or self.max_lr <= 0:
            raise ValueError('require 0 <= min_lr <= max_lr and max_lr > 0')
        if self.weight_decay < 0 or self.eps <= 0 or self.max_grad_norm <= 0:
            raise ValueError('invalid weight_decay, eps or max_grad_norm')
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError('AdamW betas must be in [0, 1)')
        if self.precision not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('precision must be float32, float16 or bfloat16')
        device = torch.device(self.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('supported devices: cpu and cuda')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested, but this Python environment has no available CUDA device')
        if self.precision != 'float32' and device.type != 'cuda':
            raise ValueError('mixed precision is supported only on CUDA; use float32 on CPU')
        if device.type == 'cuda' and self.precision == 'bfloat16':
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


def _save_atomic(state, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(config, train_tokens, val_tokens, output_dir, *, tokenizer=None, resume=None):
    """Train to max_steps (total, not additional steps), saving last.pt / best.pt.

    A step comprises grad_accum_steps equally sized microbatches. FP16 uses a
    GradScaler; overflow attempts are logged as skipped optimizer steps. CPU
    float32 resume restores RNG and optimizer states for reproducible continuation.
    """
    config.validate()
    data_hashes = {'train': _validate_data(train_tokens, config),
                   'validation': _validate_data(val_tokens, config) if val_tokens is not None else None}
    if tokenizer is not None and set(tokenizer.vocab) != set(range(config.model.vocab_size)):
        raise ValueError('tokenizer IDs must exactly cover the configured model vocabulary')
    output = Path(output_dir)
    if resume is None and any((output / name).exists() for name in ('last.pt', 'best.pt', 'metrics.jsonl')):
        raise FileExistsError('output contains a run; use --resume or a new output directory')
    state = None
    if resume is not None:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state.get('format') != 'cs336-training-v1':
            raise ValueError('resume requires a cs336-training-v1 checkpoint')
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
                checkpoint = {'format': 'cs336-training-v1', 'iteration': step,
                    'config': asdict(config), 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'best_val_loss': best_val_loss, 'data_sha256': data_hashes,
                    'tokenizer': tokenizer_state(tokenizer) if tokenizer is not None else None,
                    'train_rng': train_rng.bit_generator.state, 'val_rng': val_rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
                _save_atomic(checkpoint, output / 'last.pt')
                if improved:
                    _save_atomic(checkpoint, output / 'best.pt')
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--train-data', required=True, type=Path)
    parser.add_argument('--val-data', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--vocab', type=Path)
    parser.add_argument('--merges', type=Path)
    parser.add_argument('--special-token', action='append', default=[])
    parser.add_argument('--device')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('device', 'max_steps'):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    config = TrainConfig.from_dict(values)
    if bool(args.vocab) != bool(args.merges):
        parser.error('--vocab and --merges must be provided together')
    tokenizer = (BPE_tokenizer.from_files(args.vocab, args.merges, args.special_token)
                 if args.vocab else None)
    if tokenizer is None and args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True).get('tokenizer')
        tokenizer = BPE_tokenizer(**saved) if saved else None
    for path in (args.train_data, args.val_data):
        if path and path.with_suffix('.meta.json').exists() and tokenizer is not None:
            metadata = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
            if metadata['tokenizer_sha256'] != tokenizer_fingerprint(tokenizer):
                raise ValueError(f'tokenizer does not match tokenized dataset: {path}')
    train_tokens = np.load(args.train_data, mmap_mode='r', allow_pickle=False)
    val_tokens = np.load(args.val_data, mmap_mode='r', allow_pickle=False) if args.val_data else None
    train(config, train_tokens, val_tokens, args.output_dir, tokenizer=tokenizer, resume=args.resume)


if __name__ == '__main__':
    main()
