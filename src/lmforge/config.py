"""Typed, side-effect-free LMForge configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import math
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping


class ConfigError(ValueError):
    """A configuration error with a user-facing field path."""


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{path} must be a non-negative integer")
    return value


def _finite_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    return result


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} must be a table")
    return dict(value)


def _strict_values(
    values: object,
    cls: type,
    path: str,
) -> dict[str, Any]:
    result = _mapping(values, path)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(result) - allowed)
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(f"unknown configuration field: {prefix}{unknown[0]}")
    return result


def _required(values: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in values:
        prefix = f"{path}." if path else ""
        raise ConfigError(f"missing configuration field: {prefix}{name}")
    return values[name]


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _path(value: object, base_dir: Path, path: str) -> Path:
    result = Path(_string(value, path)).expanduser()
    if not result.is_absolute():
        result = base_dir / result
    return result.resolve(strict=False)


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        raise ConfigError(f"{path} must be an array of strings")
    return tuple(value)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1000
    context_length: int = 128
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 352
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            _positive_int(getattr(self, name), f"model.{name}")
        _finite_float(self.rope_theta, "model.rope_theta")
        if self.rope_theta <= 0:
            raise ConfigError("model.rope_theta must be positive")
        if self.d_model % self.num_heads:
            raise ConfigError("model.d_model must be divisible by model.num_heads")
        if (self.d_model // self.num_heads) % 2:
            raise ConfigError("model dimensions require an even RoPE head dimension")


@dataclass(frozen=True)
class TrainingConfig:
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

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "batch_size",
            "grad_accum_steps",
            "lr_decay_steps",
            "eval_interval",
            "eval_batches",
            "save_interval",
            "log_interval",
        ):
            _positive_int(getattr(self, name), f"training.{name}")
        _nonnegative_int(self.warmup_steps, "training.warmup_steps")
        if type(self.seed) is not int:
            raise ConfigError("training.seed must be an integer")
        for name in ("max_lr", "min_lr", "weight_decay", "beta1", "beta2", "eps", "max_grad_norm"):
            _finite_float(getattr(self, name), f"training.{name}")
        if not 0 <= self.warmup_steps < self.lr_decay_steps:
            raise ConfigError("require 0 <= training.warmup_steps < training.lr_decay_steps")
        if not 0 <= self.min_lr <= self.max_lr or self.max_lr <= 0:
            raise ConfigError("require 0 <= training.min_lr <= training.max_lr and max_lr > 0")
        if self.weight_decay < 0 or self.eps <= 0 or self.max_grad_norm <= 0:
            raise ConfigError("invalid training.weight_decay, eps or max_grad_norm")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ConfigError("training AdamW betas must be in [0, 1)")


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path
    device: str = "cpu"
    precision: str = "float32"
    deterministic: bool = False
    resume: Path | None = None

    def __post_init__(self) -> None:
        if type(self.deterministic) is not bool:
            raise ConfigError("runtime.deterministic must be a boolean")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", self.device):
            raise ConfigError("runtime.device must be cpu, cuda, or cuda:<index>")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ConfigError("runtime.precision must be float32, float16 or bfloat16")
        if self.device == "cpu" and self.precision != "float32":
            raise ConfigError("runtime requires float32 on CPU")


@dataclass(frozen=True)
class TokenizerConfig:
    vocab: Path
    merges: Path
    vocab_size: int
    special_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _positive_int(self.vocab_size, "tokenizer.vocab_size")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ConfigError("tokenizer.special_tokens must not contain duplicates")


@dataclass(frozen=True)
class PrepareDatasetConfig:
    input: Path
    output: Path


@dataclass(frozen=True)
class PrepareConfig:
    datasets: tuple[PrepareDatasetConfig, ...]

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ConfigError("prepare.datasets must contain at least one dataset")


@dataclass(frozen=True)
class DataConfig:
    train: Path
    validation: Path | None = None


@dataclass(frozen=True)
class TrainConfig:
    """Effective production config; its dictionary shape is checkpoint-stable."""

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
    device: str = "cpu"
    precision: str = "float32"
    deterministic: bool = False

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainConfig:
        data = _strict_values(values, cls, "")
        model_values = data.get("model", {})
        data["model"] = _parse_model(model_values)
        return cls(**data)

    def validate(self) -> None:
        TrainingConfig(**{item.name: getattr(self, item.name) for item in fields(TrainingConfig)})
        RuntimeConfig(
            output_dir=Path("."),
            device=self.device,
            precision=self.precision,
            deterministic=self.deterministic,
        )


@dataclass(frozen=True)
class LMForgeConfig:
    tokenizer: TokenizerConfig
    prepare: PrepareConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if self.tokenizer.vocab_size != self.model.vocab_size:
            raise ConfigError(
                "tokenizer.vocab_size must equal model.vocab_size "
                f"({self.tokenizer.vocab_size} != {self.model.vocab_size})"
            )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: str | Path,
    ) -> LMForgeConfig:
        data = _mapping(values, "configuration")
        allowed = {"tokenizer", "prepare", "data", "model", "training", "runtime"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ConfigError(f"unknown configuration field: {unknown[0]}")
        for name in allowed:
            _required(data, name, "")
        base = Path(base_dir).resolve(strict=False)
        return cls(
            tokenizer=_parse_tokenizer(data["tokenizer"], base),
            prepare=_parse_prepare(data["prepare"], base),
            data=_parse_data(data["data"], base),
            model=_parse_model(data["model"]),
            training=_parse_training(data["training"]),
            runtime=_parse_runtime(data["runtime"], base),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer": {
                "vocab": str(self.tokenizer.vocab),
                "merges": str(self.tokenizer.merges),
                "special_tokens": list(self.tokenizer.special_tokens),
                "vocab_size": self.tokenizer.vocab_size,
            },
            "prepare": {
                "datasets": [
                    {"input": str(dataset.input), "output": str(dataset.output)}
                    for dataset in self.prepare.datasets
                ]
            },
            "data": {
                "train": str(self.data.train),
                "validation": str(self.data.validation) if self.data.validation is not None else None,
            },
            "model": asdict(self.model),
            "training": asdict(self.training),
            "runtime": {
                "device": self.runtime.device,
                "precision": self.runtime.precision,
                "deterministic": self.runtime.deterministic,
                "output_dir": str(self.runtime.output_dir),
                "resume": str(self.runtime.resume) if self.runtime.resume is not None else None,
            },
        }

    def with_overrides(self, **overrides: object) -> LMForgeConfig:
        allowed = {"device", "max_steps", "output_dir", "resume"}
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ConfigError(f"unsupported override: {unknown[0]}")
        training = self.training
        runtime = self.runtime
        if "max_steps" in overrides:
            training = replace(training, max_steps=overrides["max_steps"])
        if "device" in overrides:
            runtime = replace(runtime, device=overrides["device"])
        if "output_dir" in overrides:
            output = overrides["output_dir"]
            if not isinstance(output, (str, Path)):
                raise ConfigError("override output_dir must be a path")
            runtime = replace(runtime, output_dir=Path(output).expanduser().resolve(strict=False))
        if "resume" in overrides:
            resume = overrides["resume"]
            if resume is not None and not isinstance(resume, (str, Path)):
                raise ConfigError("override resume must be a path")
            runtime = replace(
                runtime,
                resume=None if resume is None else Path(resume).expanduser().resolve(strict=False),
            )
        return replace(self, training=training, runtime=runtime)

    def to_train_config(self) -> TrainConfig:
        return TrainConfig(
            model=self.model,
            **asdict(self.training),
            device=self.runtime.device,
            precision=self.runtime.precision,
            deterministic=self.runtime.deterministic,
        )


def _parse_model(values: object) -> ModelConfig:
    data = _strict_values(values, ModelConfig, "model")
    for item in fields(ModelConfig):
        if item.name in data:
            if item.name == "rope_theta":
                data[item.name] = _finite_float(data[item.name], f"model.{item.name}")
            else:
                data[item.name] = _positive_int(data[item.name], f"model.{item.name}")
    return ModelConfig(**data)


def _parse_training(values: object) -> TrainingConfig:
    data = _strict_values(values, TrainingConfig, "training")
    integer_fields = {
        "max_steps",
        "batch_size",
        "grad_accum_steps",
        "lr_decay_steps",
        "eval_interval",
        "eval_batches",
        "save_interval",
        "log_interval",
    }
    for name in integer_fields & data.keys():
        data[name] = _positive_int(data[name], f"training.{name}")
    if "warmup_steps" in data:
        data["warmup_steps"] = _nonnegative_int(data["warmup_steps"], "training.warmup_steps")
    if "seed" in data and type(data["seed"]) is not int:
        raise ConfigError("training.seed must be an integer")
    float_fields = {"max_lr", "min_lr", "weight_decay", "beta1", "beta2", "eps", "max_grad_norm"}
    for name in float_fields & data.keys():
        data[name] = _finite_float(data[name], f"training.{name}")
    return TrainingConfig(**data)


def _parse_runtime(values: object, base_dir: Path) -> RuntimeConfig:
    data = _strict_values(values, RuntimeConfig, "runtime")
    output = _path(_required(data, "output_dir", "runtime"), base_dir, "runtime.output_dir")
    device = _string(data.get("device", "cpu"), "runtime.device")
    precision = _string(data.get("precision", "float32"), "runtime.precision")
    deterministic = data.get("deterministic", False)
    if type(deterministic) is not bool:
        raise ConfigError("runtime.deterministic must be a boolean")
    resume_value = data.get("resume")
    if resume_value is not None:
        resume_value = _path(resume_value, base_dir, "runtime.resume")
    return RuntimeConfig(
        output_dir=output,
        device=device,
        precision=precision,
        deterministic=deterministic,
        resume=resume_value,
    )


def _parse_tokenizer(values: object, base_dir: Path) -> TokenizerConfig:
    data = _strict_values(values, TokenizerConfig, "tokenizer")
    return TokenizerConfig(
        vocab=_path(_required(data, "vocab", "tokenizer"), base_dir, "tokenizer.vocab"),
        merges=_path(_required(data, "merges", "tokenizer"), base_dir, "tokenizer.merges"),
        vocab_size=_positive_int(
            _required(data, "vocab_size", "tokenizer"),
            "tokenizer.vocab_size",
        ),
        special_tokens=_string_tuple(data.get("special_tokens", []), "tokenizer.special_tokens"),
    )


def _parse_prepare(values: object, base_dir: Path) -> PrepareConfig:
    data = _strict_values(values, PrepareConfig, "prepare")
    datasets = _required(data, "datasets", "prepare")
    if not isinstance(datasets, list):
        raise ConfigError("prepare.datasets must be an array of tables")
    parsed = []
    for index, dataset in enumerate(datasets):
        path = f"prepare.datasets[{index}]"
        item = _strict_values(dataset, PrepareDatasetConfig, path)
        parsed.append(
            PrepareDatasetConfig(
                input=_path(_required(item, "input", path), base_dir, f"{path}.input"),
                output=_path(_required(item, "output", path), base_dir, f"{path}.output"),
            )
        )
    return PrepareConfig(tuple(parsed))


def _parse_data(values: object, base_dir: Path) -> DataConfig:
    data = _strict_values(values, DataConfig, "data")
    validation = data.get("validation")
    return DataConfig(
        train=_path(_required(data, "train", "data"), base_dir, "data.train"),
        validation=None if validation is None else _path(validation, base_dir, "data.validation"),
    )


def load_config(path: str | Path) -> LMForgeConfig:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            values = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot load configuration {source}: {error}") from error
    return LMForgeConfig.from_dict(values, base_dir=source.resolve().parent)
