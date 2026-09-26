from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lmforge.tokenization.serialization import save_tokenizer_files


def _console_script() -> Path:
    return Path(sys.executable).with_name("lmforge")


def _run_cli(*arguments: object, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    return subprocess.run(
        [_console_script(), *map(str, arguments)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env=env,
    )


def _project_toml(tmp_path: Path, *, extra_model: str = "") -> Path:
    path = tmp_path / "project.toml"
    path.write_text(
        f"""
[tokenizer]
vocab = "vocab.json"
merges = "merges.json"
special_tokens = ["<|endoftext|>"]
vocab_size = 257

[[prepare.datasets]]
input = "input.txt"
output = "tokens.npy"

[data]
train = "tokens.npy"

[model]
vocab_size = 257
context_length = 4
d_model = 8
num_layers = 1
num_heads = 2
d_ff = 16
rope_theta = 10000.0
{extra_model}

[training]
max_steps = 1
batch_size = 1
grad_accum_steps = 1
warmup_steps = 0
lr_decay_steps = 10
eval_interval = 2
eval_batches = 1
save_interval = 1
log_interval = 1

[runtime]
device = "cpu"
precision = "float32"
output_dir = "run"
""",
        encoding="utf-8",
    )
    return path


def test_installed_console_script_help() -> None:
    executable = _console_script()

    assert executable.is_file()
    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "{prepare,train,generate}" in result.stdout


@pytest.mark.parametrize("command", ["prepare", "train", "generate"])
def test_subcommand_help(command: str) -> None:
    result = _run_cli(command, "--help", timeout=30)

    assert result.returncode == 0, result.stderr
    assert f"lmforge {command}" in result.stdout


def test_effective_config_applies_only_documented_overrides(tmp_path: Path) -> None:
    config_path = _project_toml(tmp_path)
    output_dir = tmp_path / "override-run"
    resume = tmp_path / "checkpoint.pt"

    result = _run_cli(
        "train",
        "--config",
        config_path,
        "--device",
        "cuda",
        "--max-steps",
        7,
        "--output-dir",
        output_dir,
        "--resume",
        resume,
        "--print-effective-config",
    )

    assert result.returncode == 0, result.stderr
    effective = json.loads(result.stdout)
    assert effective["runtime"]["device"] == "cuda"
    assert effective["runtime"]["output_dir"] == str(output_dir)
    assert effective["runtime"]["resume"] == str(resume)
    assert effective["training"]["max_steps"] == 7

    rejected = _run_cli("train", "--config", config_path, "--batch-size", 4)
    assert rejected.returncode == 2
    assert "unrecognized arguments: --batch-size" in rejected.stderr


def test_invalid_config_is_a_clear_cli_error(tmp_path: Path) -> None:
    config_path = _project_toml(tmp_path, extra_model="unknown = 1")

    result = _run_cli("train", "--config", config_path, "--print-effective-config")

    assert result.returncode == 2
    assert "model.unknown" in result.stderr
    assert "Traceback" not in result.stderr


def test_real_console_prepare_train_resume_generate(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    save_tokenizer_files(
        {index: bytes([index]) for index in range(256)},
        [],
        vocab_path,
        merges_path,
    )
    (tmp_path / "input.txt").write_text("Hello world!\n" * 10, encoding="utf-8")
    config_path = _project_toml(tmp_path)

    prepared = _run_cli("prepare", "--config", config_path)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    assert (tmp_path / "tokens.npy").is_file()

    trained = _run_cli("train", "--config", config_path)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    checkpoint = tmp_path / "run" / "last.pt"
    assert checkpoint.is_file()

    resumed = _run_cli(
        "train",
        "--config",
        config_path,
        "--resume",
        checkpoint,
        "--max-steps",
        2,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr

    generated = _run_cli(
        "generate",
        "--checkpoint",
        checkpoint,
        "--prompt",
        "Hello",
        "--max-new-tokens",
        2,
        "--temperature",
        0,
        "--device",
        "cpu",
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert generated.stdout.startswith("Hello")


@pytest.mark.parametrize("module", ["prepare", "train", "generate"])
def test_legacy_module_entrypoints_delegate_to_unified_cli(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"lmforge.training.{module}", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert f"lmforge {module}" in result.stdout
