"""Frozen pre-optimization BPE path used only as a benchmark baseline.

This intentionally materializes decoded chunks and ``list[bytes]`` worker
results.  Keep application code on :mod:`cs336_basics.tokenization.train_bpe`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypeAlias

import regex as re

from cs336_basics.heap import Heap
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenization.pretokenize import PRETOKEN_PATTERN


PathLike: TypeAlias = str | os.PathLike[str]
Pair: TypeAlias = tuple[bytes, bytes]


def _legacy_pre_token(task: tuple[str, tuple[str, ...]]) -> list[bytes]:
    text, special_tokens = task
    if special_tokens:
        special_pattern = "|".join(
            re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
        )
        sections = re.split(special_pattern, text)
    else:
        sections = (text,)
    return [
        match.group().encode("utf-8")
        for section in sections
        for match in PRETOKEN_PATTERN.finditer(section)
    ]


def _materialize_chunk_inputs(
    input_path: Path,
    special_tokens: list[str],
    num_processes: int,
) -> list[tuple[str, tuple[str, ...]]]:
    file_size = input_path.stat().st_size
    if file_size == 0:
        return []
    if num_processes == 1 or not special_tokens:
        boundaries = [0, file_size]
    else:
        with input_path.open("rb") as input_file:
            boundaries = find_chunk_boundaries(input_file, num_processes, special_tokens)

    chunks: list[tuple[str, tuple[str, ...]]] = []
    with input_path.open("rb") as input_file:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            input_file.seek(start)
            text = input_file.read(end - start).replace(b"\r\n", b"\n").decode(
                "utf-8", errors="ignore"
            )
            chunks.append((text, tuple(special_tokens)))
    return chunks


def train_bpe_legacy(
    input_path: PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
) -> tuple[dict[int, bytes], list[Pair], dict[str, float | int]]:
    """Run the original materialized-corpus/in-place pair-update algorithm."""

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}")

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    chunk_inputs = _materialize_chunk_inputs(path, special_tokens, num_processes)
    if len(chunk_inputs) <= 1:
        partial_corpora = [_legacy_pre_token(task) for task in chunk_inputs]
    else:
        with mp.Pool(processes=min(num_processes, len(chunk_inputs))) as pool:
            partial_corpora = pool.map(_legacy_pre_token, chunk_inputs)
    corpus = [word for partial in partial_corpora for word in partial]
    pretokenization_seconds = time.perf_counter() - stage_start

    vocabulary = [token.encode("utf-8") for token in special_tokens]
    vocabulary.extend(bytes([value]) for value in range(256))

    stage_start = time.perf_counter()
    global_pair_count: defaultdict[Pair, int] = defaultdict(int)
    word_splits: dict[bytes, list[bytes]] = {}
    word_frequency: Counter[bytes] = Counter()
    pair_in_word_count: defaultdict[Pair, Counter[bytes]] = defaultdict(Counter)

    for word in corpus:
        word_frequency[word] += 1
        parts = [word[index : index + 1] for index in range(len(word))]
        word_splits[word] = parts
        for pair in zip(parts, parts[1:]):
            global_pair_count[pair] += 1
            pair_in_word_count[pair][word] += 1

    initial_pair_types = len(global_pair_count)
    heap = Heap(
        mode="max",
        data=[(count, pair) for pair, count in global_pair_count.items()],
    )
    pair_initialization_seconds = time.perf_counter() - stage_start

    merges: list[Pair] = []
    heap_pushes = initial_pair_types
    stale_heap_pops = 0
    affected_word_updates = 0
    stage_start = time.perf_counter()

    while len(vocabulary) < vocab_size and not heap.empty():
        selected_count, selected_pair = heap.pop()
        if global_pair_count.get(selected_pair, 0) != selected_count or selected_count <= 0:
            stale_heap_pops += 1
            continue

        left, right = selected_pair
        merged_token = left + right
        vocabulary.append(merged_token)
        merges.append(selected_pair)
        global_pair_count.pop(selected_pair, None)

        for word, indexed_count in list(pair_in_word_count[selected_pair].items()):
            if indexed_count <= 0:
                continue
            parts = word_splits[word]
            frequency = word_frequency[word]
            affected_word_updates += 1
            index = 0
            while index < len(parts) - 1:
                if (parts[index], parts[index + 1]) != selected_pair:
                    index += 1
                    continue

                if index > 0:
                    old_left_pair = (parts[index - 1], left)
                    new_left_pair = (parts[index - 1], merged_token)
                    global_pair_count[old_left_pair] -= frequency
                    global_pair_count[new_left_pair] += frequency
                    pair_in_word_count[old_left_pair][word] -= 1
                    pair_in_word_count[new_left_pair][word] += 1
                    heap.push((global_pair_count[old_left_pair], old_left_pair))
                    heap.push((global_pair_count[new_left_pair], new_left_pair))
                    heap_pushes += 2

                if index < len(parts) - 2:
                    old_right_pair = (right, parts[index + 2])
                    new_right_pair = (merged_token, parts[index + 2])
                    global_pair_count[old_right_pair] -= frequency
                    global_pair_count[new_right_pair] += frequency
                    pair_in_word_count[old_right_pair][word] -= 1
                    pair_in_word_count[new_right_pair][word] += 1
                    heap.push((global_pair_count[old_right_pair], old_right_pair))
                    heap.push((global_pair_count[new_right_pair], new_right_pair))
                    heap_pushes += 2

                parts[index] = merged_token
                del parts[index + 1]
                index += 1

        pair_in_word_count.pop(selected_pair, None)

    pair_merge_seconds = time.perf_counter() - stage_start
    metrics: dict[str, float | int] = {
        "total_seconds": time.perf_counter() - total_start,
        "pretokenization_seconds": pretokenization_seconds,
        "pair_initialization_seconds": pair_initialization_seconds,
        "pair_merge_seconds": pair_merge_seconds,
        "input_bytes": path.stat().st_size,
        "chunk_count": len(chunk_inputs),
        "unique_pretokens": len(word_frequency),
        "total_pretokens": len(corpus),
        "worker_result_items": sum(len(partial) for partial in partial_corpora),
        "initial_pair_types": initial_pair_types,
        "merges_completed": len(merges),
        "affected_word_updates": affected_word_updates,
        "heap_pushes": heap_pushes,
        "stale_heap_pops": stale_heap_pops,
        "heap_rebuilds": 0,
    }
    return dict(enumerate(vocabulary)), merges, metrics
