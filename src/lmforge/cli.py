"""Unified LMForge command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .config import ConfigError, LMForgeConfig, load_config


def _add_project_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="print canonical JSON after overrides and exit",
    )


def _load_tokenizer(config: LMForgeConfig):
    from .tokenization.tokenizer import BPE_tokenizer

    tokenizer = BPE_tokenizer.from_files(
        config.tokenizer.vocab,
        config.tokenizer.merges,
        list(config.tokenizer.special_tokens),
    )
    expected_ids = set(range(config.tokenizer.vocab_size))
    if set(tokenizer.vocab) != expected_ids:
        raise ConfigError(
            "tokenizer files must define IDs exactly covering "
            f"0..{config.tokenizer.vocab_size - 1}"
        )
    return tokenizer


def _print_effective_config(config: LMForgeConfig) -> None:
    print(json.dumps(config.to_dict(), indent=2))


def _run_prepare(args: argparse.Namespace) -> int:
    from .training.prepare import prepare_tokens

    config = load_config(args.config)
    if args.print_effective_config:
        _print_effective_config(config)
        return 0
    tokenizer = _load_tokenizer(config)
    results = [
        prepare_tokens(dataset.input, dataset.output, tokenizer)
        for dataset in config.prepare.datasets
    ]
    print(json.dumps(results, indent=2))
    return 0


def _training_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        name: getattr(args, name)
        for name in ("device", "max_steps", "output_dir", "resume")
        if getattr(args, name) is not None
    }


def _validate_dataset_metadata(config: LMForgeConfig, tokenizer) -> None:
    from .training.prepare import tokenizer_fingerprint

    for path in (config.data.train, config.data.validation):
        if path is None:
            continue
        metadata_path = path.with_suffix(".meta.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("tokenizer_sha256") != tokenizer_fingerprint(tokenizer):
                raise ConfigError(f"tokenizer does not match tokenized dataset: {path}")


def _run_train(args: argparse.Namespace) -> int:
    from .training.engine import train

    config = load_config(args.config).with_overrides(**_training_overrides(args))
    if args.print_effective_config:
        _print_effective_config(config)
        return 0
    tokenizer = _load_tokenizer(config)
    _validate_dataset_metadata(config, tokenizer)
    train_tokens = np.load(config.data.train, mmap_mode="r", allow_pickle=False)
    val_tokens = (
        np.load(config.data.validation, mmap_mode="r", allow_pickle=False)
        if config.data.validation is not None
        else None
    )
    train(
        config.to_train_config(),
        train_tokens,
        val_tokens,
        config.runtime.output_dir,
        tokenizer=tokenizer,
        resume=config.runtime.resume,
    )
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    from .training.generate import generate_text, load_model

    model, tokenizer = load_model(args.checkpoint, device=args.device)
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token=args.eos_token,
        seed=args.seed,
    )
    print(text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmforge", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="tokenize configured datasets")
    _add_project_config_arguments(prepare)
    prepare.set_defaults(handler=_run_prepare)

    train = subparsers.add_parser("train", help="train from a project configuration")
    _add_project_config_arguments(train)
    train.add_argument("--device")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--resume", type=Path)
    train.set_defaults(handler=_run_train)

    generate = subparsers.add_parser("generate", help="generate text from a checkpoint")
    generate.add_argument("--checkpoint", required=True, type=Path)
    generate.add_argument("--prompt", default="Once upon a time")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--eos-token", default="<|endoftext|>")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--device", default="cpu")
    generate.set_defaults(handler=_run_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


def legacy_main(command: str) -> int:
    """Run a legacy module entry point through the unified parser."""

    return main([command, *sys.argv[1:]])
