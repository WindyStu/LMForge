from __future__ import annotations

import csv
import json


def test_naive_trainer_freezes_tie_breaking_and_non_overlapping_merges(tmp_path) -> None:
    from lmforge.tokenization.naive_train_bpe import train_bpe_naive

    tie_path = tmp_path / "tie.txt"
    tie_path.write_text("ab ac", encoding="utf-8")
    _, tie_merges, _ = train_bpe_naive(tie_path, 257, [], num_processes=1)
    assert tie_merges == [(b"a", b"c")]

    overlap_path = tmp_path / "overlap.txt"
    overlap_path.write_text("aaaa", encoding="utf-8")
    vocab, merges, _ = train_bpe_naive(overlap_path, 258, [], num_processes=1)
    assert merges[:2] == [(b"a", b"a"), (b"aa", b"aa")]
    assert {token_id: vocab[token_id] for token_id in range(256, 258)} == {
        256: b"aa",
        257: b"aaaa",
    }


def test_naive_trainer_matches_production_and_is_deterministic(tmp_path) -> None:
    from lmforge.tokenization.naive_train_bpe import train_bpe_naive
    from lmforge.tokenization.train_bpe import train_bpe

    path = tmp_path / "corpus.txt"
    path.write_text(
        "ASCII café 🙂 aaaa<|endoftext|>second café 🙂<|endoftext|>",
        encoding="utf-8",
    )
    arguments = (path, 275, ["<|endoftext|>"])
    first_vocab, first_merges, _ = train_bpe_naive(*arguments, num_processes=1)
    second_vocab, second_merges, _ = train_bpe_naive(*arguments, num_processes=1)

    assert (first_vocab, first_merges) == (second_vocab, second_merges)
    assert (first_vocab, first_merges) == train_bpe(*arguments, num_processes=1)
    assert first_vocab[0] == b"<|endoftext|>"
    assert [first_vocab[index] for index in range(1, 257)] == [bytes([value]) for value in range(256)]


def test_phase3_benchmark_writes_raw_json_and_aggregate_csv(tmp_path) -> None:
    from lmforge.tokenization.benchmark_tokenizer import DatasetSpec, run_benchmark

    corpus = tmp_path / "ascii.txt"
    corpus.write_text("hello hello aaaa\n" * 8, encoding="utf-8")
    output_dir = tmp_path / "results"
    report = run_benchmark(
        datasets=[DatasetSpec("ascii", corpus, ())],
        output_dir=output_dir,
        vocab_size=264,
        repetitions=2,
        num_processes=1,
        throughput_bytes=1024,
        trainer="naive",
    )

    assert report["correctness"]["status"] == "passed"
    raw = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert len(raw["runs"]) == 2
    assert all(run["vocab_sha256"] and run["merges_sha256"] for run in raw["runs"])
    assert all(run["encode_mb_per_second"] > 0 for run in raw["runs"])
    assert all(run["decode_mb_per_second"] > 0 for run in raw["runs"])
    assert all(run["peak_rss_bytes"] > 0 for run in raw["runs"])

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "ascii"
    assert rows[0]["correctness_status"] == "passed"


def test_phase3_profiler_is_separate_and_reports_hotspot_shares(tmp_path) -> None:
    from lmforge.tokenization.profile_tokenizer import profile_naive_trainer

    corpus = tmp_path / "profile.txt"
    corpus.write_text("abab aaaa cababa\n" * 16, encoding="utf-8")
    output_dir = tmp_path / "profile"
    report = profile_naive_trainer(
        input_path=corpus,
        output_dir=output_dir,
        vocab_size=264,
        special_tokens=[],
        num_processes=1,
    )

    assert report["hotspots"]
    assert report["stage_percentages"]["repeated_pair_count_percent"] >= 0
    assert report["stage_percentages"]["repeated_vocabulary_scan_percent"] >= 0
    assert report["python_allocation"]["peak_traced_bytes"] > 0
    assert (output_dir / "naive.prof").is_file()
    assert (output_dir / "profile.json").is_file()
