from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import TypeAlias

import regex as re

from .pretokenize import PRETOKEN_PATTERN
from .serialization import load_tokenizer_files


PathLike: TypeAlias = str | os.PathLike[str]
Pair: TypeAlias = tuple[bytes, bytes]


class BPE_tokenizer:
    """Byte-level BPE tokenizer with deterministic merge-rank semantics."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        if len(set(vocab.values())) != len(vocab):
            raise ValueError("vocab contains duplicate byte strings")
        if len(set(merges)) != len(merges):
            raise ValueError("merges contains duplicate pairs")

        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(special_tokens or [])

        next_id = max(self.vocab, default=-1) + 1
        existing_tokens = set(self.vocab.values())
        for special_token in self.special_tokens:
            encoded = special_token.encode("utf-8")
            if encoded not in existing_tokens:
                self.vocab[next_id] = encoded
                existing_tokens.add(encoded)
                next_id += 1

        self.reverse_vocab = {token: token_id for token_id, token in self.vocab.items()}
        self._special_pattern = self._compile_special_pattern(self.special_tokens)

        missing_results = [
            left + right
            for left, right in merges
            if left + right not in self.reverse_vocab
        ]
        if missing_results:
            raise ValueError("each merge result must exist in vocab")

    @staticmethod
    def _compile_special_pattern(special_tokens: list[str]) -> re.Pattern[str] | None:
        if not special_tokens:
            return None
        alternatives = sorted(set(special_tokens), key=len, reverse=True)
        return re.compile("|".join(re.escape(token) for token in alternatives))

    @classmethod
    def from_files(
        cls,
        vocab_filepath: PathLike,
        merges_filepath: PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BPE_tokenizer":
        """Construct a tokenizer from versioned files written by BPE training."""

        vocab, merges = load_tokenizer_files(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _merge_pair(parts: list[bytes], selected_pair: Pair) -> list[bytes]:
        merged: list[bytes] = []
        index = 0
        while index < len(parts):
            if index + 1 < len(parts) and (parts[index], parts[index + 1]) == selected_pair:
                merged.append(parts[index] + parts[index + 1])
                index += 2
            else:
                merged.append(parts[index])
                index += 1
        return merged

    def _encode_pretoken(self, token: bytes) -> list[int]:
        parts = [token[index : index + 1] for index in range(len(token))]
        while len(parts) > 1:
            ranked_pairs = (
                (self.merge_ranks[pair], pair)
                for pair in zip(parts, parts[1:])
                if pair in self.merge_ranks
            )
            best = min(ranked_pairs, default=None)
            if best is None:
                break
            parts = self._merge_pair(parts, best[1])

        try:
            return [self.reverse_vocab[part] for part in parts]
        except KeyError as error:
            raise ValueError(f"vocab has no token for byte sequence {error.args[0]!r}") from error

    def _pre_tokenize(self, text: str) -> Iterator[tuple[bytes, bool]]:
        if self._special_pattern is None:
            for match in PRETOKEN_PATTERN.finditer(text):
                yield match.group().encode("utf-8"), False
            return

        cursor = 0
        for special_match in self._special_pattern.finditer(text):
            for match in PRETOKEN_PATTERN.finditer(text, cursor, special_match.start()):
                yield match.group().encode("utf-8"), False
            yield special_match.group().encode("utf-8"), True
            cursor = special_match.end()
        for match in PRETOKEN_PATTERN.finditer(text, cursor):
            yield match.group().encode("utf-8"), False

    def encode(self, text: str) -> list[int]:
        encoded_ids: list[int] = []
        for token, is_special in self._pre_tokenize(text):
            if is_special:
                encoded_ids.append(self.reverse_vocab[token])
            else:
                encoded_ids.extend(self._encode_pretoken(token))
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        try:
            encoded = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID {error.args[0]}") from error
        return encoded.decode("utf-8", errors="replace")


Tokenizer = BPE_tokenizer
