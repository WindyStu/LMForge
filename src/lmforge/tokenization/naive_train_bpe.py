"""Deterministic full-rescan BPE trainer used as a Phase 3 baseline only."""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import TypeAlias

from .pretokenize import count_pretokens

Pair: TypeAlias = tuple[bytes, bytes]


@dataclass(frozen=True)
class NaiveBPEMetrics:
    total_seconds: float
    pretokenization_seconds: float
    repeated_pair_count_seconds: float
    repeated_vocabulary_scan_seconds: float
    input_bytes: int
    unique_pretokens: int
    total_pretokens: int
    worker_result_items: int
    pair_count_scans: int
    word_merge_scans: int
    pair_observations: int
    merges_completed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


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


def train_bpe_naive(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[Pair], NaiveBPEMetrics]:
    """Train BPE by recounting every live adjacent pair before every merge.

    This deliberately simple ``O(merges * live_symbol_count)`` implementation
    is a correctness oracle and profiler target. Application code must continue
    to use :func:`lmforge.tokenization.train_bpe.train_bpe`.
    """

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must not contain duplicates")
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    word_frequency, pretoken_stats = count_pretokens(path, special_tokens, num_processes=num_processes)
    pretokenization_seconds = time.perf_counter() - stage_start

    vocabulary = [token.encode("utf-8") for token in special_tokens]
    vocabulary.extend(bytes([value]) for value in range(256))
    word_splits = {word: [word[index : index + 1] for index in range(len(word))] for word in word_frequency}
    merges: list[Pair] = []
    repeated_pair_count_seconds = 0.0
    repeated_vocabulary_scan_seconds = 0.0
    pair_count_scans = 0
    word_merge_scans = 0
    pair_observations = 0

    while len(vocabulary) < vocab_size:
        stage_start = time.perf_counter()
        pair_counts: Counter[Pair] = Counter()
        for word, parts in word_splits.items():
            frequency = word_frequency[word]
            local_counts = Counter(pairwise(parts))
            pair_observations += sum(local_counts.values())
            for pair, count in local_counts.items():
                pair_counts[pair] += count * frequency
        repeated_pair_count_seconds += time.perf_counter() - stage_start
        pair_count_scans += 1
        if not pair_counts:
            break

        selected_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(selected_pair)
        vocabulary.append(selected_pair[0] + selected_pair[1])

        stage_start = time.perf_counter()
        for word, parts in word_splits.items():
            word_splits[word] = _merge_pair(parts, selected_pair)
            word_merge_scans += 1
        repeated_vocabulary_scan_seconds += time.perf_counter() - stage_start

    metrics = NaiveBPEMetrics(
        total_seconds=time.perf_counter() - total_start,
        pretokenization_seconds=pretokenization_seconds,
        repeated_pair_count_seconds=repeated_pair_count_seconds,
        repeated_vocabulary_scan_seconds=repeated_vocabulary_scan_seconds,
        input_bytes=pretoken_stats.input_bytes,
        unique_pretokens=pretoken_stats.unique_pretokens,
        total_pretokens=pretoken_stats.total_pretokens,
        worker_result_items=pretoken_stats.worker_result_items,
        pair_count_scans=pair_count_scans,
        word_merge_scans=word_merge_scans,
        pair_observations=pair_observations,
        merges_completed=len(merges),
    )
    return dict(enumerate(vocabulary)), merges, metrics
