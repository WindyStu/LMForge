"""Reproducible baseline-versus-optimized BPE benchmark and profiler target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import psutil

from .legacy_train_bpe import train_bpe_legacy
from .train_bpe import train_bpe_with_metrics


Result = dict[str, Any]


def _artifact_hash(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> str:
    digest = hashlib.sha256()
    for token_id, token in sorted(vocab.items()):
        digest.update(token_id.to_bytes(8, "big", signed=False))
        digest.update(len(token).to_bytes(8, "big", signed=False))
        digest.update(token)
    digest.update(b"\xffMERGES\xff")
    for left, right in merges:
        for part in (left, right):
            digest.update(len(part).to_bytes(8, "big", signed=False))
            digest.update(part)
    return digest.hexdigest()


def run_worker(
    implementation: str,
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> Result:
    """Execute one training run in the current process (the profiler entry point)."""

    if implementation == "baseline":
        vocab, merges, metrics = train_bpe_legacy(
            input_path,
            vocab_size,
            special_tokens,
            num_processes=num_processes,
        )
    elif implementation == "optimized":
        result = train_bpe_with_metrics(
            input_path,
            vocab_size,
            special_tokens,
            num_processes=num_processes,
        )
        vocab, merges, metrics = result.vocab, result.merges, result.metrics.to_dict()
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    return {
        "implementation": implementation,
        "processes": num_processes,
        "artifact_sha256": _artifact_hash(vocab, merges),
        "vocab_entries": len(vocab),
        "merge_entries": len(merges),
        "metrics": metrics,
    }


def _process_tree_rss(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    total = 0
    for member in processes:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _run_isolated(
    implementation: str,
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> Result:
    command = [
        sys.executable,
        "-m",
        "lmforge.tokenization.benchmark_bpe",
        "--worker",
        "--implementation",
        implementation,
        "--input",
        str(input_path),
        "--vocab-size",
        str(vocab_size),
        "--num-processes",
        str(num_processes),
    ]
    for special_token in special_tokens:
        command.extend(("--special-token", special_token))

    start = time.perf_counter()
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process = psutil.Process(child.pid)
    peak_rss_bytes = 0
    while child.poll() is None:
        peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss(process))
        time.sleep(0.01)
    stdout, stderr = child.communicate()
    peak_rss_bytes = max(peak_rss_bytes, _process_tree_rss(process))
    wall_seconds = time.perf_counter() - start

    if child.returncode != 0:
        raise RuntimeError(
            f"benchmark worker failed ({implementation}, processes={num_processes}):\n{stderr}"
        )
    output_lines = [line for line in stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("benchmark worker produced no JSON result")
    result = json.loads(output_lines[-1])
    result["wall_seconds"] = wall_seconds
    result["peak_rss_bytes"] = peak_rss_bytes
    return result


def _median_metric(results: list[Result], field: str) -> float:
    return float(statistics.median(float(result[field]) for result in results))


def _median_nested_metric(results: list[Result], field: str) -> float:
    return float(statistics.median(float(result["metrics"][field]) for result in results))


def _reduction_percent(baseline: float, optimized: float) -> float:
    if baseline == 0:
        return 0.0
    return round((baseline - optimized) / baseline * 100.0, 4)


def summarize_results(results: list[Result]) -> dict[str, Result]:
    """Summarize medians and compute positive-is-better improvement percentages."""

    grouped: defaultdict[tuple[int, str], list[Result]] = defaultdict(list)
    for result in results:
        grouped[(int(result["processes"]), str(result["implementation"]))].append(result)

    summaries: dict[str, Result] = {}
    process_counts = sorted({processes for processes, _ in grouped})
    for processes in process_counts:
        baseline = grouped.get((processes, "baseline"), [])
        optimized = grouped.get((processes, "optimized"), [])
        if not baseline or not optimized:
            continue

        baseline_wall = _median_metric(baseline, "wall_seconds")
        optimized_wall = _median_metric(optimized, "wall_seconds")
        baseline_rss = _median_metric(baseline, "peak_rss_bytes")
        optimized_rss = _median_metric(optimized, "peak_rss_bytes")
        baseline_items = _median_nested_metric(baseline, "worker_result_items")
        optimized_items = _median_nested_metric(optimized, "worker_result_items")
        summaries[str(processes)] = {
            "baseline_median_wall_seconds": baseline_wall,
            "optimized_median_wall_seconds": optimized_wall,
            "runtime_improvement_percent": _reduction_percent(baseline_wall, optimized_wall),
            "baseline_median_peak_rss_bytes": baseline_rss,
            "optimized_median_peak_rss_bytes": optimized_rss,
            "peak_rss_reduction_percent": _reduction_percent(baseline_rss, optimized_rss),
            "baseline_median_worker_result_items": baseline_items,
            "optimized_median_worker_result_items": optimized_items,
            "worker_result_item_reduction_percent": _reduction_percent(
                baseline_items, optimized_items
            ),
        }
    return summaries


def run_suite(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    process_counts: list[int],
    repeat: int,
    implementations: list[str],
) -> Result:
    results: list[Result] = []
    for processes in process_counts:
        for repeat_index in range(1, repeat + 1):
            for implementation in implementations:
                result = _run_isolated(
                    implementation,
                    input_path,
                    vocab_size,
                    special_tokens,
                    processes,
                )
                result["repeat"] = repeat_index
                results.append(result)
                print(
                    f"{implementation:9s} p={processes} run={repeat_index}: "
                    f"{result['wall_seconds']:.3f}s, "
                    f"peak={result['peak_rss_bytes'] / 1024**2:.1f} MiB",
                    file=sys.stderr,
                    flush=True,
                )

    hashes_by_process: defaultdict[int, set[str]] = defaultdict(set)
    for result in results:
        hashes_by_process[int(result["processes"])].add(result["artifact_sha256"])
    mismatches = {key: values for key, values in hashes_by_process.items() if len(values) != 1}
    if mismatches:
        raise RuntimeError(f"baseline/optimized tokenizer outputs differ: {mismatches}")

    return {
        "configuration": {
            "input": str(input_path.resolve()),
            "input_bytes": input_path.stat().st_size,
            "vocab_size": vocab_size,
            "special_tokens": special_tokens,
            "process_counts": process_counts,
            "repeat": repeat,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "results": results,
        "summary": summarize_results(results),
    }


def _parse_process_counts(value: str) -> list[int]:
    process_counts = [int(item) for item in value.split(",")]
    if not process_counts or any(item < 1 for item in process_counts):
        raise argparse.ArgumentTypeError("process counts must be positive integers")
    return process_counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument(
        "--implementation",
        choices=("baseline", "optimized", "both"),
        default="both",
    )
    parser.add_argument("--process-counts", type=_parse_process_counts, default=[1, 4])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    if arguments.worker:
        if arguments.implementation == "both":
            raise SystemExit("--worker requires one implementation")
        print(
            json.dumps(
                run_worker(
                    arguments.implementation,
                    arguments.input,
                    arguments.vocab_size,
                    arguments.special_token,
                    arguments.num_processes,
                ),
                sort_keys=True,
            )
        )
        return

    if arguments.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    implementations = (
        ["baseline", "optimized"]
        if arguments.implementation == "both"
        else [arguments.implementation]
    )
    report = run_suite(
        arguments.input,
        arguments.vocab_size,
        arguments.special_token,
        arguments.process_counts,
        arguments.repeat,
        implementations,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
        print(f"report: {arguments.output}", file=sys.stderr)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
