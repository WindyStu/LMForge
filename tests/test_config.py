from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

import lmforge.config as config_module

VALID_CONFIG = """
[tokenizer]
vocab = "tokenizer/vocab.json"
merges = "tokenizer/merges.json"
special_tokens = ["<|endoftext|>"]
vocab_size = 8

[[prepare.datasets]]
input = "raw/train.txt"
output = "tokens/train.npy"

[[prepare.datasets]]
input = "raw/validation.txt"
output = "tokens/validation.npy"

[data]
train = "tokens/train.npy"
validation = "tokens/validation.npy"

[model]
vocab_size = 8
context_length = 4
d_model = 8
num_layers = 1
num_heads = 2
d_ff = 16
rope_theta = 10000.0

[training]
max_steps = 4
batch_size = 2
grad_accum_steps = 2
warmup_steps = 0
lr_decay_steps = 10
eval_interval = 2
eval_batches = 2
save_interval = 2
log_interval = 1

[runtime]
device = "cpu"
precision = "float32"
output_dir = "runs/small"
"""


def _write_config(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_config_module_exists() -> None:
    assert importlib.util.find_spec("lmforge.config") is not None


def test_config_public_api_exists() -> None:
    expected = {"ConfigError", "LMForgeConfig", "ModelConfig", "TrainConfig", "load_config"}

    assert expected <= set(dir(config_module))


def test_config_module_has_no_cli_or_runtime_dependencies() -> None:
    source_path = Path(config_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imports.isdisjoint({"argparse", "numpy", "torch"})


def test_training_module_reexports_canonical_config_types() -> None:
    from lmforge.training.train import ModelConfig as TrainingModelConfig
    from lmforge.training.train import TrainConfig as TrainingTrainConfig

    assert TrainingModelConfig is config_module.ModelConfig
    assert TrainingTrainConfig is config_module.TrainConfig


def test_valid_config_round_trip_and_relative_paths(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    config = config_module.load_config(path)
    canonical = config.to_dict()
    restored = config_module.LMForgeConfig.from_dict(canonical, base_dir=tmp_path)

    assert restored == config
    assert config.tokenizer.vocab == tmp_path / "tokenizer/vocab.json"
    assert config.prepare.datasets[1].output == tmp_path / "tokens/validation.npy"
    assert config.data.train == tmp_path / "tokens/train.npy"
    assert config.runtime.output_dir == tmp_path / "runs/small"
    assert canonical["model"]["d_model"] == 8
    assert canonical["runtime"]["device"] == "cpu"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (VALID_CONFIG + "\n[unknown]\nvalue = 1\n", "unknown"),
        (VALID_CONFIG.replace("d_model = 8", "d_model = 8\nunknown = 1"), "model.unknown"),
        (VALID_CONFIG.replace("context_length = 4", "context_length = true"), "model.context_length"),
    ],
)
def test_unknown_fields_and_wrong_types_are_rejected(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    with pytest.raises(config_module.ConfigError, match=message):
        config_module.load_config(_write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            VALID_CONFIG.replace("d_model = 8", "d_model = 10").replace("num_heads = 2", "num_heads = 3"),
            "divisible",
        ),
        (
            VALID_CONFIG.replace("d_model = 8", "d_model = 6"),
            "even RoPE head dimension",
        ),
        (
            VALID_CONFIG.replace("batch_size = 2", "batch_size = 0"),
            "training.batch_size",
        ),
        (
            VALID_CONFIG.replace("warmup_steps = 0", "warmup_steps = 10"),
            "warmup_steps",
        ),
    ],
)
def test_invalid_dimensions_and_training_ranges_are_rejected(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    with pytest.raises(config_module.ConfigError, match=message):
        config_module.load_config(_write_config(tmp_path, text))


def test_invalid_precision_device_combination_is_rejected(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace('precision = "float32"', 'precision = "float16"')

    with pytest.raises(config_module.ConfigError, match="float32 on CPU"):
        config_module.load_config(_write_config(tmp_path, text))


def test_attention_backend_defaults_to_reference_and_round_trips(tmp_path: Path) -> None:
    default_config = config_module.load_config(_write_config(tmp_path))
    sdpa_config = config_module.load_config(
        _write_config(
            tmp_path,
            VALID_CONFIG.replace(
                "rope_theta = 10000.0",
                'rope_theta = 10000.0\nattention_backend = "sdpa"',
            ),
        )
    )

    assert default_config.model.attention_backend == "reference"
    assert sdpa_config.model.attention_backend == "sdpa"
    assert sdpa_config.to_train_config().model.attention_backend == "sdpa"
    assert sdpa_config.to_dict()["model"]["attention_backend"] == "sdpa"


def test_invalid_attention_backend_is_rejected(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace(
        "rope_theta = 10000.0",
        'rope_theta = 10000.0\nattention_backend = "flash"',
    )

    with pytest.raises(config_module.ConfigError, match="attention_backend"):
        config_module.load_config(_write_config(tmp_path, text))


def test_tokenizer_and_model_vocab_sizes_must_match(tmp_path: Path) -> None:
    text = VALID_CONFIG.replace("vocab_size = 8", "vocab_size = 9", 1)

    with pytest.raises(config_module.ConfigError, match="tokenizer.vocab_size.*model.vocab_size"):
        config_module.load_config(_write_config(tmp_path, text))


def test_allowlisted_overrides_are_immutable_and_derive_train_config(tmp_path: Path) -> None:
    config = config_module.load_config(_write_config(tmp_path))
    output_dir = tmp_path / "override-run"
    resume = tmp_path / "last.pt"

    effective = config.with_overrides(
        device="cuda",
        max_steps=7,
        output_dir=output_dir,
        resume=resume,
    )
    train_config = effective.to_train_config()

    assert config.runtime.device == "cpu"
    assert config.training.max_steps == 4
    assert effective.runtime.device == "cuda"
    assert effective.runtime.output_dir == output_dir
    assert effective.runtime.resume == resume
    assert train_config.model == effective.model
    assert train_config.max_steps == 7
    assert train_config.device == "cuda"
    assert train_config.precision == "float32"


def test_deterministic_runtime_defaults_off_and_reaches_training(tmp_path: Path) -> None:
    default_config = config_module.load_config(_write_config(tmp_path))
    enabled_config = config_module.load_config(
        _write_config(
            tmp_path,
            VALID_CONFIG.replace(
                'precision = "float32"',
                'precision = "float32"\ndeterministic = true',
            ),
        )
    )

    assert default_config.runtime.deterministic is False
    assert default_config.to_train_config().deterministic is False
    assert enabled_config.runtime.deterministic is True
    assert enabled_config.to_train_config().deterministic is True
    assert enabled_config.to_dict()["runtime"]["deterministic"] is True


def test_compile_model_defaults_off_and_reaches_training(tmp_path: Path) -> None:
    default_config = config_module.load_config(_write_config(tmp_path))
    enabled_config = config_module.load_config(
        _write_config(
            tmp_path,
            VALID_CONFIG.replace(
                'precision = "float32"',
                'precision = "float32"\ncompile_model = true',
            ),
        )
    )

    assert default_config.runtime.compile_model is False
    assert default_config.to_train_config().compile_model is False
    assert enabled_config.runtime.compile_model is True
    assert enabled_config.to_train_config().compile_model is True
    assert enabled_config.to_dict()["runtime"]["compile_model"] is True


def test_unsupported_override_is_rejected(tmp_path: Path) -> None:
    config = config_module.load_config(_write_config(tmp_path))

    with pytest.raises(config_module.ConfigError, match="unsupported override: batch_size"):
        config.with_overrides(batch_size=4)
