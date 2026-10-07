"""Profiler-only entry point for the Phase 3 full-rescan naive trainer."""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import pstats
import tracemalloc
from pathlib import Path
from typing import Any

from .naive_train_bpe import train_bpe_naive


def profile_naive_trainer(
    *,
    input_path: Path,
    output_dir: Path,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "naive.prof"
    profiler = cProfile.Profile()
    profiler.enable()
    _, _, metrics = train_bpe_naive(input_path, vocab_size, special_tokens, num_processes=num_processes)
    profiler.disable()
    profiler.dump_stats(profile_path)

    tracemalloc.start()
    train_bpe_naive(input_path, vocab_size, special_tokens, num_processes=num_processes)
    current_traced, peak_traced = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = pstats.Stats(profiler)
    hotspots = []
    for (filename, line, function), values in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[
        :30
    ]:
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        hotspots.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "self_seconds": self_seconds,
                "cumulative_seconds": cumulative_seconds,
                "cumulative_percent": cumulative_seconds / max(stats.total_tt, 1e-12) * 100,
            }
        )
    allocations = []
    for statistic in snapshot.statistics("lineno")[:20]:
        frame = statistic.traceback[0]
        allocations.append(
            {
                "file": frame.filename,
                "line": frame.lineno,
                "live_bytes": statistic.size,
                "live_blocks": statistic.count,
            }
        )

    def cumulative_for(*patterns: str) -> float:
        return sum(
            values[3]
            for (filename, _, function), values in stats.stats.items()
            if any(pattern in f"{filename}:{function}" for pattern in patterns)
        )

    metric_values = metrics.to_dict()
    total = max(float(metric_values["total_seconds"]), 1e-12)
    report = {
        "schema": "lmforge-tokenizer-profile-v1",
        "input": str(input_path.resolve()),
        "configuration": {"vocab_size": vocab_size, "special_tokens": special_tokens, "num_processes": num_processes},
        "metrics": metric_values,
        "stage_percentages": {
            "pretokenization_percent": float(metric_values["pretokenization_seconds"]) / total * 100,
            "repeated_pair_count_percent": float(metric_values["repeated_pair_count_seconds"]) / total * 100,
            "repeated_vocabulary_scan_percent": float(metric_values["repeated_vocabulary_scan_seconds"]) / total * 100,
        },
        "python_allocation": {
            "current_traced_bytes": current_traced,
            "peak_traced_bytes": peak_traced,
            "largest_live_allocations": allocations,
            "separate_from_cpu_profile": True,
            "limitation": "tracemalloc attributes live Python allocations; process-tree RSS is measured by the benchmark",
        },
        "focus_evidence": {
            "repeated_pair_count": {
                "seconds": metric_values["repeated_pair_count_seconds"],
                "full_scans": metric_values["pair_count_scans"],
                "pair_observations": metric_values["pair_observations"],
            },
            "repeated_vocabulary_scan": {
                "seconds": metric_values["repeated_vocabulary_scan_seconds"],
                "word_scans": metric_values["word_merge_scans"],
            },
            "pair_merge_cprofile_seconds": cumulative_for("naive_train_bpe.py:_merge_pair"),
            "pretokenization_cprofile_seconds": cumulative_for(
                "pretokenize.py:count_pretokens", "pretokenize.py:_count_chunk"
            ),
            "file_read_cprofile_seconds": cumulative_for("read"),
            "multiprocessing_pickle_cprofile_seconds": cumulative_for("multiprocessing", "pickle"),
            "multiprocessing_active": num_processes > 1,
        },
        "cprofile_total_seconds": stats.total_tt,
        "hotspots": hotspots,
    }
    (output_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "hotspots.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(hotspots[0]) if hotspots else ["function"])
        writer.writeheader()
        writer.writerows(hotspots)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3-p3-01/profile"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--num-processes", type=int, default=1)
    arguments = parser.parse_args()
    profile_naive_trainer(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        vocab_size=arguments.vocab_size,
        special_tokens=arguments.special_token,
        num_processes=arguments.num_processes,
    )


if __name__ == "__main__":
    main()
