from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

from ..heap import Heap
from .pretokenize import PretokenizationStats, count_pretokens


Pair: TypeAlias = tuple[bytes, bytes]


@dataclass(frozen=True)
class BPETrainingMetrics:
    """Stage timings and operation counts for reproducible profiling."""

    total_seconds: float
    pretokenization_seconds: float
    pair_initialization_seconds: float
    pair_merge_seconds: float
    input_bytes: int
    chunk_count: int
    unique_pretokens: int
    total_pretokens: int
    worker_result_items: int
    initial_pair_types: int
    merges_completed: int
    affected_word_updates: int
    heap_pushes: int
    stale_heap_pops: int
    heap_rebuilds: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BPETrainingResult:
    vocab: dict[int, bytes]
    merges: list[Pair]
    metrics: BPETrainingMetrics


def _validate_arguments(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> Path:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must not contain duplicates")
    minimum_vocab_size = 256 + len(special_tokens)
    if vocab_size < minimum_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {minimum_vocab_size} "
            "(one entry per special token plus 256 byte tokens)"
        )
    return path


def _adjacent_pair_counts(parts: list[bytes]) -> Counter[Pair]:
    return Counter(zip(parts, parts[1:]))


def _merge_pair(parts: list[bytes], selected_pair: Pair) -> list[bytes]:
    """Merge all non-overlapping occurrences of ``selected_pair`` in one word."""

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


def _pop_live_pair(
    heap: Heap,
    global_pair_count: dict[Pair, int],
) -> tuple[Pair | None, int]:
    stale_pops = 0
    while not heap.empty():
        count, pair = heap.pop()
        if count > 0 and global_pair_count.get(pair, 0) == count:
            return pair, stale_pops
        stale_pops += 1
    return None, stale_pops


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
):
    result = train_bpe_with_metrics(
        input_path,
        vocab_size,
        special_tokens,
        num_processes=num_processes,
    )
    return result.vocab, result.merges


def train_bpe_with_metrics(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
):
    path = _validate_arguments(input_path, vocab_size, special_tokens, num_processes)
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    word_frequency, pretoken_stats = count_pretokens(
        path,
        special_tokens,
        num_processes=num_processes,
    )
    pretokenization_seconds = time.perf_counter() - stage_start

    vocabulary = [token.encode("utf-8") for token in special_tokens]
    vocabulary.extend(bytes([value]) for value in range(256))
    merges: list[Pair] = []

    stage_start = time.perf_counter()
    word_splits: dict[bytes, list[bytes]] = {
        word: [word[index : index + 1] for index in range(len(word))]
        for word in word_frequency
    }
    global_pair_count: defaultdict[Pair, int] = defaultdict(int)
    pair_in_word_count: defaultdict[Pair, dict[bytes, int]] = defaultdict(dict)

    for word, parts in word_splits.items():
        frequency = word_frequency[word]
        for pair, local_count in _adjacent_pair_counts(parts).items():
            pair_in_word_count[pair][word] = local_count
            global_pair_count[pair] += local_count * frequency

    initial_pair_types = len(global_pair_count)
    heap = Heap(
        mode="max",
        data=[(count, pair) for pair, count in global_pair_count.items() if count > 0],
    )
    heap_pushes = initial_pair_types
    pair_initialization_seconds = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    affected_word_updates = 0
    stale_heap_pops = 0
    heap_rebuilds = 0

    while len(vocabulary) < vocab_size:
        selected_pair, discarded = _pop_live_pair(heap, global_pair_count)
        stale_heap_pops += discarded
        if selected_pair is None:
            break

        affected_words = list(pair_in_word_count.get(selected_pair, ()))
        changed_pairs: set[Pair] = set()

        for word in affected_words:
            old_parts = word_splits[word]
            old_counts = _adjacent_pair_counts(old_parts)
            new_parts = _merge_pair(old_parts, selected_pair)
            new_counts = _adjacent_pair_counts(new_parts)
            frequency = word_frequency[word]
            affected_word_updates += 1

            for pair in old_counts.keys() | new_counts.keys():
                old_local_count = old_counts.get(pair, 0)
                new_local_count = new_counts.get(pair, 0)
                if old_local_count == new_local_count:
                    continue

                global_pair_count[pair] += (new_local_count - old_local_count) * frequency
                if new_local_count:
                    pair_in_word_count[pair][word] = new_local_count
                else:
                    words_for_pair = pair_in_word_count.get(pair)
                    if words_for_pair is not None:
                        words_for_pair.pop(word, None)
                        if not words_for_pair:
                            pair_in_word_count.pop(pair, None)
                changed_pairs.add(pair)

            word_splits[word] = new_parts

        # Push only the final value after every affected word has been updated.
        # Older heap entries remain and are discarded lazily by _pop_live_pair.
        for pair in changed_pairs:
            current_count = global_pair_count.get(pair, 0)
            if current_count > 0:
                heap.push((current_count, pair))
                heap_pushes += 1
            else:
                global_pair_count.pop(pair, None)

        # Lazy invalidation avoids O(number_of_pairs) heap repair on each merge,
        # but stale entries must not grow without bound on long training runs.
        rebuild_threshold = max(50_000, 4 * len(global_pair_count))
        if len(heap) > rebuild_threshold:
            live_entries = [
                (count, pair)
                for pair, count in global_pair_count.items()
                if count > 0
            ]
            heap = Heap(mode="max", data=live_entries)
            heap_pushes += len(live_entries)
            heap_rebuilds += 1

        merges.append(selected_pair)
        vocabulary.append(selected_pair[0] + selected_pair[1])

    pair_merge_seconds = time.perf_counter() - stage_start
    total_seconds = time.perf_counter() - total_start
    metrics = BPETrainingMetrics(
        total_seconds=total_seconds,
        pretokenization_seconds=pretokenization_seconds,
        pair_initialization_seconds=pair_initialization_seconds,
        pair_merge_seconds=pair_merge_seconds,
        input_bytes=pretoken_stats.input_bytes,
        chunk_count=pretoken_stats.chunk_count,
        unique_pretokens=pretoken_stats.unique_pretokens,
        total_pretokens=pretoken_stats.total_pretokens,
        worker_result_items=pretoken_stats.worker_result_items,
        initial_pair_types=initial_pair_types,
        merges_completed=len(merges),
        affected_word_updates=affected_word_updates,
        heap_pushes=heap_pushes,
        stale_heap_pops=stale_heap_pops,
        heap_rebuilds=heap_rebuilds,
    )
    return BPETrainingResult(dict(enumerate(vocabulary)), merges, metrics)
