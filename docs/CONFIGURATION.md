# Configuration and Unified CLI Contract

## Purpose

Phase 1 task P1-06 establishes one strict TOML configuration boundary and one
installed `lmforge` command without changing tokenizer algorithms, model
mathematics, the training step lifecycle, or the checkpoint schema.

## Configuration ownership

`lmforge.config` owns immutable dataclasses, TOML loading, strict type and field
validation, cross-field validation, documented overrides, and canonical
serialization. It must not import `argparse`, construct a model, open a dataset,
start training, or modify global random state.

The configuration types are:

- `ModelConfig`: model dimensions and RoPE parameters;
- `TrainingConfig`: optimizer, schedule, evaluation, checkpoint cadence, and
  seed values declared under `[training]`;
- `RuntimeConfig`: device, precision, output directory, and optional resume
  checkpoint;
- `TokenizerConfig`: vocab and merges paths, special tokens, and declared vocab
  size;
- `PrepareDatasetConfig`: one UTF-8 input and token-array output pair;
- `DataConfig`: training and optional validation token-array paths;
- `LMForgeConfig`: the complete project configuration;
- `TrainConfig`: the derived production view consumed by the existing training
  loop.

`TrainConfig` retains the exact nested model and flat training/runtime field
shape currently serialized in `cs336-training-v1` checkpoints. Existing Python
imports from `lmforge.training.train` remain valid through re-exported names.

## TOML schema

A single project TOML is shared by `prepare` and `train`:

```toml
[tokenizer]
vocab = "../../artifacts/tokenizer/vocab.json"
merges = "../../artifacts/tokenizer/merges.json"
special_tokens = ["<|endoftext|>"]
vocab_size = 1000

[[prepare.datasets]]
input = "../../data/tinystories/train.txt"
output = "../../artifacts/data/train.npy"

[[prepare.datasets]]
input = "../../data/tinystories/validation.txt"
output = "../../artifacts/data/validation.npy"

[data]
train = "../../artifacts/data/train.npy"
validation = "../../artifacts/data/validation.npy"

[model]
vocab_size = 1000
context_length = 128
d_model = 128
num_layers = 2
num_heads = 4
d_ff = 352
rope_theta = 10000.0

[training]
max_steps = 1000
batch_size = 2
grad_accum_steps = 8
max_lr = 0.0003
min_lr = 0.00003
warmup_steps = 50
lr_decay_steps = 1000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8
max_grad_norm = 1.0
eval_interval = 100
eval_batches = 10
save_interval = 100
log_interval = 10
seed = 42

[runtime]
device = "cpu"
precision = "float32"
output_dir = "../../artifacts/runs/tinystories-small"
```

Relative paths resolve against the TOML file's directory, not the process
working directory. Unknown top-level sections and unknown nested fields are
errors. Boolean values are not accepted as integers. Canonical serialization is
JSON-compatible and deterministic in field order.

## Validation

Pure configuration validation covers at least:

- positive vocabulary size, context length, model dimensions, layer count,
  batch size, gradient accumulation, steps, and intervals;
- `d_model` divisible by `num_heads`;
- an even RoPE head dimension;
- positive finite RoPE theta and finite training hyperparameters;
- `0 <= warmup_steps < lr_decay_steps`;
- valid learning-rate, AdamW beta, epsilon, weight-decay, and gradient-norm
  ranges;
- precision in `float32`, `float16`, or `bfloat16`;
- device type limited to CPU or CUDA, with non-float32 precision limited to
  CUDA;
- declared tokenizer vocab size equal to model vocab size.

Hardware capability checks, actual tokenizer loading, tokenizer ID coverage,
dataset metadata fingerprints, and CUDA availability are CLI/production
boundary responsibilities rather than pure configuration responsibilities.

## Unified command

`pyproject.toml` installs one console script:

```text
lmforge = lmforge.cli:main
```

Supported commands are:

```text
lmforge prepare --config PATH
lmforge train --config PATH [--device ...] [--max-steps ...]
              [--output-dir ...] [--resume ...]
lmforge generate --checkpoint PATH [generation options]
```

Prepare processes every `[[prepare.datasets]]` entry with the configured
tokenizer. Train loads the configured tokenizer and memory-mapped arrays,
verifies tokenizer IDs and dataset fingerprints, derives `TrainConfig`, and
calls the existing `train()` production API. Generate treats the checkpoint as
the sole model configuration and tokenizer source.

Prepare and train support `--print-effective-config`, which prints canonical
JSON after strict loading and allowlisted overrides, then exits without opening
tokenizer or dataset files or starting training. Train overrides are limited to
`device`, `max_steps`, `output_dir`, and `resume`; arbitrary key/value overrides
are not supported.

Expected configuration and input errors produce concise CLI errors and a
nonzero exit status. Unexpected internal exceptions are not silently swallowed.

## Compatibility

`python -m lmforge.training.prepare`, `python -m lmforge.training.train`, and
`python -m lmforge.training.generate` remain executable as thin wrappers around
the unified parser. They do not retain separate argument or configuration
parsers. Prepare and train therefore migrate to the project TOML interface,
while generation preserves its checkpoint-driven arguments.

The training production API, numerical behavior, effective `TrainConfig`
dictionary, and `cs336-training-v1` checkpoint format remain unchanged.

## Test-driven implementation

The config commit is implemented first from failing tests for round-trip,
unknown fields, strict types, model dimensions, schedule relationships,
precision/device combinations, path resolution, and tokenizer/model vocab
agreement.

The CLI commit follows from failing subprocess tests for the installed console
script, help commands, effective-config output, allowlisted overrides, clear
configuration failures, the real prepare/train/resume/generate workflow, and
legacy module wrappers.

Completion requires `uv sync`, all four requested help commands, the complete
pytest suite with no regression below the 71-pass baseline, a clean explained
Git status, and confirmation that local coordination and AI files are absent
from tracked paths.
