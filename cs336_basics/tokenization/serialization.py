from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


FORMAT_VERSION = "cs336-bpe-v1"


def _validate_vocab(vocab: dict[int, bytes]) -> None:
    if any(not isinstance(token_id, int) or token_id < 0 for token_id in vocab):
        raise ValueError("vocab IDs must be non-negative integers")
    if any(not isinstance(token, bytes) for token in vocab.values()):
        raise TypeError("vocab values must be bytes")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("vocab contains duplicate byte strings")


def _validate_merges(merges: list[tuple[bytes, bytes]]) -> None:
    for merge in merges:
        if (
            not isinstance(merge, tuple)
            or len(merge) != 2
            or not all(isinstance(part, bytes) for part in merge)
        ):
            raise TypeError("each merge must be a tuple containing two bytes values")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error.msg}") from error


def _decode_hex(value: object, *, context: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a hexadecimal string")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"invalid hexadecimal bytes for {context}") from error


def save_tokenizer_files(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike[str],
    merges_path: str | os.PathLike[str],
) -> None:
    """Save arbitrary byte tokens losslessly using a versioned JSON/hex format."""

    _validate_vocab(vocab)
    _validate_merges(merges)
    vocab_destination = Path(vocab_path)
    merges_destination = Path(merges_path)
    vocab_destination.parent.mkdir(parents=True, exist_ok=True)
    merges_destination.parent.mkdir(parents=True, exist_ok=True)

    vocab_payload = {
        "format": FORMAT_VERSION,
        "tokens": [
            {"id": token_id, "bytes": token.hex()}
            for token_id, token in sorted(vocab.items())
        ],
    }
    merges_payload = {
        "format": FORMAT_VERSION,
        "merges": [[left.hex(), right.hex()] for left, right in merges],
    }
    vocab_destination.write_text(
        json.dumps(vocab_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    merges_destination.write_text(
        json.dumps(merges_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_tokenizer_files(
    vocab_path: str | os.PathLike[str],
    merges_path: str | os.PathLike[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Load and validate files emitted by :func:`save_tokenizer_files`."""

    vocab_source = Path(vocab_path)
    merges_source = Path(merges_path)
    vocab_payload = _read_json(vocab_source)
    merges_payload = _read_json(merges_source)

    if not isinstance(vocab_payload, dict) or vocab_payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported vocabulary format in {vocab_source}")
    if not isinstance(merges_payload, dict) or merges_payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported merges format in {merges_source}")

    token_entries = vocab_payload.get("tokens")
    merge_entries = merges_payload.get("merges")
    if not isinstance(token_entries, list):
        raise ValueError("vocabulary field 'tokens' must be a list")
    if not isinstance(merge_entries, list):
        raise ValueError("merges field 'merges' must be a list")

    vocab: dict[int, bytes] = {}
    for index, entry in enumerate(token_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"vocabulary entry {index} must be an object")
        token_id = entry.get("id")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"vocabulary entry {index} has an invalid ID")
        if token_id in vocab:
            raise ValueError(f"duplicate vocabulary ID {token_id}")
        vocab[token_id] = _decode_hex(entry.get("bytes"), context=f"token ID {token_id}")

    merges: list[tuple[bytes, bytes]] = []
    for index, entry in enumerate(merge_entries):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(f"merge entry {index} must contain exactly two values")
        merges.append(
            (
                _decode_hex(entry[0], context=f"merge {index} left side"),
                _decode_hex(entry[1], context=f"merge {index} right side"),
            )
        )

    _validate_vocab(vocab)
    _validate_merges(merges)
    return vocab, merges
