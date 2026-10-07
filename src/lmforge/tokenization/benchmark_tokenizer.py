"""Standalone Phase 3 tokenizer training/encode/decode benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from .naive_train_bpe import train_bpe_naive
from .tokenizer import BPE_tokenizer
from .train_bpe import train_bpe_with_metrics

BUILTIN_FIXTURES: dict[str, tuple[str, tuple[str, ...]]] = {
    "ascii": ("The quick brown fox jumps. aaaa abab\n" * 8, ()),
    "unicode": ("café naïve 東京 привет aaaa\n" * 8, ()),
    "emoji": ("🙂🚀🌍 👩‍💻 🏳️‍🌈 aaaa\n" * 8, ()),
    "special_token": (
        "before<|endoftext|>after<|endoftext|> aaaa\n" * 8,
        ("<|endoftext|>",),
    ),
    "overlapping_special_token": (
        "x<|endoftext|><|endoftext|>y<|endoftext|> aaaa\n" * 8,
        ("<|endoftext|>", "<|endoftext|><|endoftext|>"),
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    special_tokens: tuple[str, ...]


def _vocab_hash(vocab: dict[int, bytes]) -> str:
    digest = hashlib.sha256()
    for token_id, token in sorted(vocab.items()):
        digest.update(token_id.to_bytes(8, "big"))
        digest.update(len(token).to_bytes(8, "big"))
        digest.update(token)
    return digest.hexdigest()


def _merges_hash(merges: list[tuple[bytes, bytes]]) -> str:
    digest = hashlib.sha256()
    for left, right in merges:
        for part in (left, right):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker(
    dataset: DatasetSpec,
    vocab_size: int,
    num_processes: int,
    throughput_bytes: int,
    trainer: str,
) -> dict[str, Any]:
    train_start = time.perf_counter()
    if trainer == "naive":
        vocab, merges, metrics_object = train_bpe_naive(
            dataset.path, vocab_size, list(dataset.special_tokens), num_processes=num_processes
        )
        metrics = metrics_object.to_dict()
    else:
        result = train_bpe_with_metrics(
            dataset.path, vocab_size, list(dataset.special_tokens), num_processes=num_processes
        )
        vocab, merges, metrics = result.vocab, result.merges, result.metrics.to_dict()
    train_seconds = time.perf_counter() - train_start

    source_text = dataset.path.read_text(encoding="utf-8")
    source_bytes = len(source_text.encode("utf-8"))
    repeats = max(1, (throughput_bytes + max(source_bytes, 1) - 1) // max(source_bytes, 1))
    text = source_text * repeats
    measured_bytes = len(text.encode("utf-8"))
    tokenizer = BPE_tokenizer(vocab, merges, list(dataset.special_tokens))
    tokenizer.encode(text)

    encode_start = time.perf_counter()
    ids = tokenizer.encode(text)
    encode_seconds = time.perf_counter() - encode_start
    decode_start = time.perf_counter()
    decoded = tokenizer.decode(ids)
    decode_seconds = time.perf_counter() - decode_start
    special_atomic = all(
        tokenizer.encode(token) == [tokenizer.reverse_vocab[token.encode("utf-8")]] for token in dataset.special_tokens
    )
    return {
        "dataset": dataset.name,
        "trainer": trainer,
        "input_path": str(dataset.path.resolve()),
        "input_bytes": dataset.path.stat().st_size,
        "input_sha256": _file_hash(dataset.path),
        "measured_bytes": measured_bytes,
        "train_seconds": train_seconds,
        "train_mb_per_second": dataset.path.stat().st_size / 1_000_000 / train_seconds,
        "encode_seconds": encode_seconds,
        "encode_mb_per_second": measured_bytes / 1_000_000 / encode_seconds,
        "decode_seconds": decode_seconds,
        "decode_mb_per_second": measured_bytes / 1_000_000 / decode_seconds,
        "encoded_tokens": len(ids),
        "vocab_entries": len(vocab),
        "merge_entries": len(merges),
        "vocab_sha256": _vocab_hash(vocab),
        "merges_sha256": _merges_hash(merges),
        "correctness": {
            "roundtrip": decoded == text,
            "special_tokens_atomic": special_atomic,
            "status": "passed" if decoded == text and special_atomic else "failed",
        },
        "metrics": metrics,
    }


def _tree_rss(process: psutil.Process) -> int:
    members = [process]
    try:
        members.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _isolated_run(
    dataset: DatasetSpec,
    vocab_size: int,
    num_processes: int,
    throughput_bytes: int,
    trainer: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "lmforge.tokenization.benchmark_tokenizer",
        "--worker",
        "--dataset-name",
        dataset.name,
        "--input",
        str(dataset.path),
        "--vocab-size",
        str(vocab_size),
        "--num-processes",
        str(num_processes),
        "--throughput-bytes",
        str(throughput_bytes),
        "--trainer",
        trainer,
    ]
    for token in dataset.special_tokens:
        command.extend(("--special-token", token))
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process = psutil.Process(child.pid)
    peak = 0
    while child.poll() is None:
        peak = max(peak, _tree_rss(process))
        time.sleep(0.01)
    stdout, stderr = child.communicate()
    if child.returncode:
        raise RuntimeError(stderr)
    result = json.loads([line for line in stdout.splitlines() if line][-1])
    result["peak_rss_bytes"] = max(peak, int(result.get("peak_rss_bytes", 0)))
    return result


def _median(runs: list[dict[str, Any]], field: str) -> float:
    return float(statistics.median(float(run[field]) for run in runs))


def _mean(runs: list[dict[str, Any]], field: str) -> float:
    return float(statistics.mean(float(run[field]) for run in runs))


def _stdev(runs: list[dict[str, Any]], field: str) -> float:
    values = [float(run[field]) for run in runs]
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _git_identity() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_benchmark(
    *,
    datasets: list[DatasetSpec],
    output_dir: Path,
    vocab_size: int,
    repetitions: int,
    num_processes: int,
    throughput_bytes: int,
    trainer: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainers = ["naive", "production"] if trainer == "both" else [trainer]
    runs: list[dict[str, Any]] = []
    for dataset in datasets:
        for repetition in range(1, repetitions + 1):
            ordered_trainers = trainers if repetition % 2 else list(reversed(trainers))
            for selected_trainer in ordered_trainers:
                result = _isolated_run(dataset, vocab_size, num_processes, throughput_bytes, selected_trainer)
                result["repetition"] = repetition
                runs.append(result)

    summary: list[dict[str, Any]] = []
    correctness_passed = True
    for dataset in datasets:
        reference = train_bpe_with_metrics(
            dataset.path, vocab_size, list(dataset.special_tokens), num_processes=num_processes
        )
        reference_hashes = (_vocab_hash(reference.vocab), _merges_hash(reference.merges))
        for selected_trainer in trainers:
            selected = [run for run in runs if run["dataset"] == dataset.name and run["trainer"] == selected_trainer]
            hashes = {(run["vocab_sha256"], run["merges_sha256"]) for run in selected}
            passed = (
                len(hashes) == 1
                and next(iter(hashes)) == reference_hashes
                and all(run["correctness"]["status"] == "passed" for run in selected)
            )
            correctness_passed &= passed
            summary.append(
                {
                    "dataset": dataset.name,
                    "trainer": selected_trainer,
                    "runs": len(selected),
                    "train_seconds_median": _median(selected, "train_seconds"),
                    "train_seconds_mean": _mean(selected, "train_seconds"),
                    "train_seconds_stdev": _stdev(selected, "train_seconds"),
                    "train_mb_per_second_median": _median(selected, "train_mb_per_second"),
                    "encode_mb_per_second_median": _median(selected, "encode_mb_per_second"),
                    "encode_mb_per_second_stdev": _stdev(selected, "encode_mb_per_second"),
                    "decode_mb_per_second_median": _median(selected, "decode_mb_per_second"),
                    "decode_mb_per_second_stdev": _stdev(selected, "decode_mb_per_second"),
                    "peak_rss_bytes_median": _median(selected, "peak_rss_bytes"),
                    "peak_rss_bytes_stdev": _stdev(selected, "peak_rss_bytes"),
                    "vocab_sha256": selected[0]["vocab_sha256"],
                    "merges_sha256": selected[0]["merges_sha256"],
                    "correctness_status": "passed" if passed else "failed",
                }
            )

    comparisons: list[dict[str, Any]] = []
    if trainer == "both":
        for dataset in datasets:
            naive = next(row for row in summary if row["dataset"] == dataset.name and row["trainer"] == "naive")
            production = next(
                row for row in summary if row["dataset"] == dataset.name and row["trainer"] == "production"
            )
            naive_seconds = float(naive["train_seconds_median"])
            production_seconds = float(production["train_seconds_median"])
            naive_rss = float(naive["peak_rss_bytes_median"])
            production_rss = float(production["peak_rss_bytes_median"])
            comparisons.append(
                {
                    "dataset": dataset.name,
                    "naive_train_seconds_median": naive_seconds,
                    "production_train_seconds_median": production_seconds,
                    "train_speedup": naive_seconds / production_seconds,
                    "naive_train_mb_per_second_median": naive["train_mb_per_second_median"],
                    "production_train_mb_per_second_median": production["train_mb_per_second_median"],
                    "naive_peak_rss_bytes_median": naive_rss,
                    "production_peak_rss_bytes_median": production_rss,
                    "peak_rss_reduction_percent": (naive_rss - production_rss) / naive_rss * 100,
                    "vocab_hash_match": naive["vocab_sha256"] == production["vocab_sha256"],
                    "merges_hash_match": naive["merges_sha256"] == production["merges_sha256"],
                    "correctness_status": (
                        "passed"
                        if naive["correctness_status"] == production["correctness_status"] == "passed"
                        else "failed"
                    ),
                }
            )
    report = {
        "schema": "lmforge-tokenizer-benchmark-v2",
        "configuration": {
            "vocab_size": vocab_size,
            "repetitions": repetitions,
            "num_processes": num_processes,
            "throughput_bytes": throughput_bytes,
            "trainer": trainer,
            "datasets": [asdict(dataset) for dataset in datasets],
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "git": _git_identity(),
        },
        "correctness": {"status": "passed" if correctness_passed else "failed"},
        "runs": runs,
        "summary": summary,
        "comparisons": comparisons,
    }
    (output_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summary[0]) if summary else ["dataset"])
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(comparisons[0]) if comparisons else ["dataset"])
        writer.writeheader()
        writer.writerows(comparisons)
    return report


def _safe_prefix(source: Path, destination: Path, byte_limit: int) -> None:
    with source.open("rb") as input_file:
        data = input_file.read(byte_limit)
    text = data.decode("utf-8", errors="ignore")
    destination.write_text(text, encoding="utf-8", newline="")


def prepare_datasets(
    output_dir: Path, tiny_stories: Path | None, openwebtext: Path | None, subset_bytes: int
) -> list[DatasetSpec]:
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[DatasetSpec] = []
    for name, (text, specials) in BUILTIN_FIXTURES.items():
        path = dataset_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8", newline="")
        datasets.append(DatasetSpec(name, path, specials))
    for name, source in (("tinystories_subset", tiny_stories), ("openwebtext_subset", openwebtext)):
        if source is not None:
            path = dataset_dir / f"{name}.txt"
            _safe_prefix(source, path, subset_bytes)
            datasets.append(DatasetSpec(name, path, ("<|endoftext|>",)))
    return datasets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3-p3-01/benchmark"))
    parser.add_argument("--tiny-stories", type=Path)
    parser.add_argument("--openwebtext", type=Path)
    parser.add_argument("--subset-bytes", type=int, default=1_048_576)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--throughput-bytes", type=int, default=1_048_576)
    parser.add_argument("--trainer", choices=("naive", "production", "both"), default="naive")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-name", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--special-token", action="append", default=[], help=argparse.SUPPRESS)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.worker:
        if arguments.trainer == "both":
            raise SystemExit("--worker requires one concrete trainer")
        result = _worker(
            DatasetSpec(arguments.dataset_name, arguments.input, tuple(arguments.special_token)),
            arguments.vocab_size,
            arguments.num_processes,
            arguments.throughput_bytes,
            arguments.trainer,
        )
        print(json.dumps(result, sort_keys=True))
        return
    datasets = prepare_datasets(
        arguments.output_dir, arguments.tiny_stories, arguments.openwebtext, arguments.subset_bytes
    )
    report = run_benchmark(
        datasets=datasets,
        output_dir=arguments.output_dir,
        vocab_size=arguments.vocab_size,
        repetitions=arguments.repetitions,
        num_processes=arguments.num_processes,
        throughput_bytes=arguments.throughput_bytes,
        trainer=arguments.trainer,
    )
    if report["correctness"]["status"] != "passed":
        raise SystemExit("tokenizer correctness gate failed")


if __name__ == "__main__":
    main()
