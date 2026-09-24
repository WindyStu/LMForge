from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path

import regex as re


PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _merge_pair(parts: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    merged: list[bytes] = []
    index = 0
    while index < len(parts):
        if index + 1 < len(parts) and (parts[index], parts[index + 1]) == pair:
            merged.append(parts[index] + parts[index + 1])
            index += 2
        else:
            merged.append(parts[index])
            index += 1
    return merged


def _slow_reference(
    text: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    sections = re.split(special_pattern, text) if special_pattern else [text]
    word_frequency = Counter(
        match.group().encode("utf-8")
        for section in sections
        for match in PATTERN.finditer(section)
    )

    vocabulary = [token.encode("utf-8") for token in special_tokens]
    vocabulary.extend(bytes([value]) for value in range(256))
    word_splits = {word: [word[i : i + 1] for i in range(len(word))] for word in word_frequency}
    merges: list[tuple[bytes, bytes]] = []

    while len(vocabulary) < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, parts in word_splits.items():
            frequency = word_frequency[word]
            pair_counts.update(
                {pair: count * frequency for pair, count in Counter(zip(parts, parts[1:])).items()}
            )
        if not pair_counts:
            break

        selected_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(selected_pair)
        vocabulary.append(selected_pair[0] + selected_pair[1])
        for word, parts in word_splits.items():
            word_splits[word] = _merge_pair(parts, selected_pair)

    return dict(enumerate(vocabulary)), merges


def test_public_bpe_optimization_api_exists() -> None:
    train_module = importlib.import_module("cs336_basics.tokenization.train_bpe")
    serialization_module = importlib.import_module("cs336_basics.tokenization.serialization")
    tokenizer_module = importlib.import_module("cs336_basics.tokenization.tokenizer")

    assert hasattr(train_module, "train_bpe")
    assert hasattr(train_module, "train_bpe_with_metrics")
    assert hasattr(serialization_module, "save_tokenizer_files")
    assert hasattr(serialization_module, "load_tokenizer_files")
    assert hasattr(tokenizer_module, "BPE_tokenizer")


def test_optimized_training_matches_slow_reference(tmp_path) -> None:
    from cs336_basics.tokenization.train_bpe import train_bpe

    corpus = "xaba xaba xababa<|endoftext|>aaaa aaaa\n"
    input_path = tmp_path / "corpus.txt"
    input_path.write_text(corpus, encoding="utf-8")

    expected_vocab, expected_merges = _slow_reference(
        corpus,
        vocab_size=265,
        special_tokens=["<|endoftext|>"],
    )
    actual_vocab, actual_merges = train_bpe(
        input_path,
        vocab_size=265,
        special_tokens=["<|endoftext|>"],
        num_processes=1,
    )

    assert actual_vocab == expected_vocab
    assert actual_merges == expected_merges


def test_parallel_training_matches_single_process(tmp_path) -> None:
    from cs336_basics.tokenization.train_bpe import train_bpe

    input_path = tmp_path / "documents.txt"
    input_path.write_text(
        "first story abababa<|endoftext|>"
        "second story cababa<|endoftext|>"
        "third story abacab\n",
        encoding="utf-8",
    )
    arguments = {
        "input_path": input_path,
        "vocab_size": 272,
        "special_tokens": ["<|endoftext|>"],
    }

    assert train_bpe(**arguments, num_processes=2) == train_bpe(**arguments, num_processes=1)


def test_benchmark_baseline_and_optimized_are_equivalent(tmp_path) -> None:
    from cs336_basics.tokenization.legacy_train_bpe import train_bpe_legacy
    from cs336_basics.tokenization.train_bpe import train_bpe

    input_path = tmp_path / "corpus.txt"
    input_path.write_text("abab ababa cab<|endoftext|>abab\n", encoding="utf-8")
    arguments = (input_path, 268, ["<|endoftext|>"])

    legacy_vocab, legacy_merges, _ = train_bpe_legacy(*arguments, num_processes=1)
    optimized_vocab, optimized_merges = train_bpe(*arguments, num_processes=1)

    assert (legacy_vocab, legacy_merges) == (optimized_vocab, optimized_merges)


def test_serialization_roundtrip_preserves_arbitrary_bytes(tmp_path) -> None:
    from cs336_basics.tokenization.serialization import (
        load_tokenizer_files,
        save_tokenizer_files,
    )

    vocab = {0: b"\x00", 1: b"\xff", 2: b"\x00\xff"}
    merges = [(b"\x00", b"\xff")]
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"

    save_tokenizer_files(vocab, merges, vocab_path, merges_path)

    assert load_tokenizer_files(vocab_path, merges_path) == (vocab, merges)


def test_tokenizer_from_files_can_encode_and_decode(tmp_path) -> None:
    from cs336_basics.tokenization.serialization import save_tokenizer_files
    from cs336_basics.tokenization.tokenizer import BPE_tokenizer

    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"ab"
    merges = [(b"a", b"b")]
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    save_tokenizer_files(vocab, merges, vocab_path, merges_path)

    tokenizer = BPE_tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=["<|endoftext|>"],
    )
    text = "ab<|endoftext|>ab"

    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode("ab") == [256]


def test_benchmark_summary_reports_runtime_and_memory_percentages() -> None:
    from cs336_basics.tokenization.benchmark_bpe import summarize_results

    results = [
        {
            "implementation": "baseline",
            "processes": 1,
            "wall_seconds": 10.0,
            "peak_rss_bytes": 1_000,
            "metrics": {"worker_result_items": 1_000},
        },
        {
            "implementation": "optimized",
            "processes": 1,
            "wall_seconds": 4.0,
            "peak_rss_bytes": 600,
            "metrics": {"worker_result_items": 250},
        },
    ]

    summary = summarize_results(results)["1"]

    assert summary["runtime_improvement_percent"] == 60.0
    assert summary["peak_rss_reduction_percent"] == 40.0
    assert summary["worker_result_item_reduction_percent"] == 75.0


def test_wsl_profile_script_uses_scalene_23_arguments() -> None:
    script_path = Path(__file__).parents[1] / "run_bpe_profile_wsl.sh"
    script = script_path.read_text(encoding="utf-8")

    assert "--cpu \\" not in script
    assert "--html \\" not in script
    assert "--memory \\" not in script
    assert '_p1_scalene.json" \\' in script
    assert "    --- \\" in script


def test_wsl_profile_script_uses_scalene_run_subcommand() -> None:
    script_path = Path(__file__).parents[1] / "run_bpe_profile_wsl.sh"
    script = script_path.read_text(encoding="utf-8")

    assert "uv run scalene \\\n    run \\" in script
