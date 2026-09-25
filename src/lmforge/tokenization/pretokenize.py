"""Parallel pre-token counting for BPE training.

Workers return a ``Counter`` of unique UTF-8 pre-tokens instead of a materialized
``list[bytes]``.  This keeps repeated words out of multiprocessing messages and
out of the parent process' corpus representation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import regex as re

from ..pretokenization_example import find_chunk_boundaries


PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

PathLike: TypeAlias = str | os.PathLike[str]
ChunkTask: TypeAlias = tuple[str, int, int, tuple[str, ...]]


@dataclass(frozen=True)
class PretokenizationStats:
    """Size information used by the benchmark report."""

    input_bytes: int
    chunk_count: int
    unique_pretokens: int
    total_pretokens: int
    worker_result_items: int


def _special_pattern(special_tokens: tuple[str, ...]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    alternatives = sorted(special_tokens, key=len, reverse=True)
    return re.compile("|".join(re.escape(token) for token in alternatives))


def count_pretokens_in_text(
    text: str,
    special_tokens: tuple[str, ...] | list[str],
) -> Counter[bytes]:
    """Count regex pre-tokens after excluding all configured special tokens."""

    specials = tuple(special_tokens)
    pattern = _special_pattern(specials)
    sections = pattern.split(text) if pattern is not None else (text,)
    return Counter(
        match.group().encode("utf-8")
        for section in sections
        for match in PRETOKEN_PATTERN.finditer(section)
    )


def _count_chunk(task: ChunkTask) -> Counter[bytes]:
    """Picklable worker entry point; each worker reads only its own byte range."""

    input_path, start, end, special_tokens = task
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        text = input_file.read(end - start).replace(b"\r\n", b"\n").decode(
            "utf-8", errors="ignore"
        )
    return count_pretokens_in_text(text, special_tokens)


def _chunk_tasks(
    input_path: Path,
    special_tokens: tuple[str, ...],
    num_processes: int,
) -> list[ChunkTask]:
    file_size = input_path.stat().st_size
    if file_size == 0:
        return []

    # Without a document delimiter, an arbitrary byte boundary can split both a
    # UTF-8 code point and a regex pre-token.  Falling back to one chunk preserves
    # correctness; callers can still use parallelism when a delimiter is supplied.
    if num_processes == 1 or not special_tokens:
        boundaries = [0, file_size]
    else:
        with input_path.open("rb") as input_file:
            boundaries = find_chunk_boundaries(
                input_file,
                desired_num_chunks=num_processes,
                split_special_tokens=list(special_tokens),
            )

    path_string = str(input_path)
    return [
        (path_string, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]


def count_pretokens(
    input_path: PathLike,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
) -> tuple[Counter[bytes], PretokenizationStats]:
    """Return global pre-token frequencies and benchmark-oriented size stats."""

    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    specials = tuple(special_tokens)
    tasks = _chunk_tasks(path, specials, num_processes)
    combined: Counter[bytes] = Counter()
    worker_result_items = 0
    if len(tasks) <= 1:
        for task in tasks:
            counts = _count_chunk(task)
            combined.update(counts)
            worker_result_items += len(counts)
    else:
        with mp.Pool(processes=min(num_processes, len(tasks))) as pool:
            for counts in pool.imap_unordered(_count_chunk, tasks):
                combined.update(counts)
                worker_result_items += len(counts)

    stats = PretokenizationStats(
        input_bytes=path.stat().st_size,
        chunk_count=len(tasks),
        unique_pretokens=len(combined),
        total_pretokens=sum(combined.values()),
        worker_result_items=worker_result_items,
    )
    return combined, stats
