"""Pure aggregation and traceable inference benchmark outputs."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "ttft_seconds",
    "prefill_seconds",
    "decode_seconds",
    "decode_tokens_per_second",
    "end_to_end_seconds",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "cache_allocated_bytes",
    "cache_cuda_allocated_delta_bytes",
    "cache_theoretical_bytes",
)
IMPLEMENTATIONS = ("generate_naive", "generate_with_kv_cache")


def _identity(row, *, paired=False):
    config = dict(row["configuration"])
    if paired:
        config.pop("implementation")
    return json.dumps(config, sort_keys=True)


def _stats(values):
    return {"mean": statistics.fmean(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def _valid(rows, required_runs):
    return (
        len(rows) == required_runs
        and {row["run"] for row in rows} == set(range(1, required_runs + 1))
        and len({row.get("worker_pid") for row in rows}) == required_runs
        and all(row.get("status") == "ok" and row.get("internal_parity") for row in rows)
        and all(row.get("git", {}).get("dirty") is False for row in rows)
        and all(
            len({row.get(field) for row in rows}) == 1 and rows[0].get(field)
            for field in ("input_sha256", "weights_sha256", "output_sha256")
        )
        and len({row.get("git", {}).get("sha") for row in rows}) == 1
        and len({json.dumps(row.get("environment"), sort_keys=True) for row in rows}) == 1
        and all(all(field in row.get("measurement", {}) for field in METRICS) for row in rows)
    )


def aggregate(rows, *, required_runs=3):
    if required_runs < 3:
        raise ValueError("publication requires at least three independent runs")
    groups = defaultdict(list)
    pairs = defaultdict(list)
    for row in rows:
        groups[_identity(row)].append(row)
        pairs[_identity(row, paired=True)].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        config = json.loads(key)
        record = {
            **config,
            "runs": len(group),
            "successful_runs": sum(row["status"] == "ok" for row in group),
            "readme_eligible": _valid(group, required_runs),
        }
        for metric in METRICS:
            values = [
                row["measurement"][metric]
                for row in group
                if row["status"] == "ok" and metric in row.get("measurement", {})
            ]
            if values:
                record.update({metric + "_" + name: value for name, value in _stats(values).items()})
        summary.append(record)
    comparison = []
    for key, group in sorted(pairs.items()):
        by_impl = {
            impl: sorted(
                [row for row in group if row["configuration"]["implementation"] == impl], key=lambda row: row["run"]
            )
            for impl in IMPLEMENTATIONS
        }
        valid = all(_valid(values, required_runs) for values in by_impl.values())
        valid = valid and len({row.get("worker_pid") for row in group}) == 2 * required_runs
        valid = valid and all(
            len({row.get(field) for row in group}) == 1 for field in ("input_sha256", "weights_sha256", "output_sha256")
        )
        valid = valid and len({row.get("git", {}).get("sha") for row in group}) == 1
        valid = valid and len({json.dumps(row.get("environment"), sort_keys=True) for row in group}) == 1
        record = {**json.loads(key), "readme_eligible": bool(valid), "runs_per_implementation": required_runs}
        if valid:
            naive, cached = (by_impl[impl] for impl in IMPLEMENTATIONS)
            for name, metric in (
                ("decode_speedup", "decode_seconds"),
                ("end_to_end_speedup", "end_to_end_seconds"),
                ("ttft_speedup", "ttft_seconds"),
            ):
                values = [
                    a["measurement"][metric] / b["measurement"][metric] for a, b in zip(naive, cached, strict=True)
                ]
                record.update({name + "_" + stat: value for stat, value in _stats(values).items()})
        comparison.append(record)
    return {"raw": rows, "summary": summary, "comparison": comparison}


def _csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def write_results(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    derived = aggregate(report["results"], required_runs=report.get("independent_runs", 3))
    raw = []
    for row in derived["raw"]:
        flat = {key: value for key, value in row.items() if key not in ("configuration", "measurement")}
        flat.update(row["configuration"])
        flat.update(row.get("measurement", {}))
        raw.append(flat)
    _csv(directory / "raw-runs.csv", raw)
    _csv(directory / "summary.csv", derived["summary"])
    _csv(directory / "comparison.csv", derived["comparison"])
    (directory / "summary.json").write_text(
        json.dumps({key: value for key, value in derived.items() if key != "raw"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return derived


def tables(derived):
    lookup = {(row["prompt_length"], row["generated_tokens"], row["implementation"]): row for row in derived["summary"]}

    def cell(row, metric, scale=1.0, digits=2):
        return f"{row[metric + '_mean'] / scale:.{digits}f} ± {row[metric + '_stdev'] / scale:.{digits}f}"

    sections = {
        "decode": [
            "| Prompt | New tokens | Naive decode tok/s | Cached decode tok/s | Decode speedup | Naive E2E s | Cached E2E s | E2E speedup |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ],
        "phases": [
            "| Prompt | New tokens | Naive TTFT ms | Cached TTFT ms | Naive prefill ms | Cached prefill ms | Naive decode s | Cached decode s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ],
        "memory": [
            "| Prompt | New tokens | Naive allocated MiB | Cached allocated MiB | Naive reserved MiB | Cached reserved MiB | Cache storage MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ],
    }
    for comparison in sorted(derived["comparison"], key=lambda row: (row["prompt_length"], row["generated_tokens"])):
        if not comparison["readme_eligible"]:
            continue
        prompt, generated = comparison["prompt_length"], comparison["generated_tokens"]
        naive, cached = (lookup[(prompt, generated, impl)] for impl in IMPLEMENTATIONS)
        sections["decode"].append(
            "| "
            + " | ".join(
                [
                    str(prompt),
                    str(generated),
                    cell(naive, "decode_tokens_per_second"),
                    cell(cached, "decode_tokens_per_second"),
                    cell(comparison, "decode_speedup") + "x",
                    cell(naive, "end_to_end_seconds"),
                    cell(cached, "end_to_end_seconds"),
                    cell(comparison, "end_to_end_speedup") + "x",
                ]
            )
            + " |"
        )
        sections["phases"].append(
            "| "
            + " | ".join(
                [
                    str(prompt),
                    str(generated),
                    cell(naive, "ttft_seconds", 0.001),
                    cell(cached, "ttft_seconds", 0.001),
                    cell(naive, "prefill_seconds", 0.001),
                    cell(cached, "prefill_seconds", 0.001),
                    cell(naive, "decode_seconds"),
                    cell(cached, "decode_seconds"),
                ]
            )
            + " |"
        )
        sections["memory"].append(
            "| "
            + " | ".join(
                [
                    str(prompt),
                    str(generated),
                    cell(naive, "peak_allocated_bytes", 2**20),
                    cell(cached, "peak_allocated_bytes", 2**20),
                    cell(naive, "peak_reserved_bytes", 2**20),
                    cell(cached, "peak_reserved_bytes", 2**20),
                    cell(cached, "cache_allocated_bytes", 2**20),
                ]
            )
            + " |"
        )
    return {name: "\n".join(lines) for name, lines in sections.items()}


def charts(derived, output):
    # Import plotting libraries only in publication, never in GPU workers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "lmforge-phase4",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = [row for row in derived["summary"] if row["readme_eligible"]]
    colors = {
        prompt: color
        for prompt, color in zip(
            sorted({row["prompt_length"] for row in summary}), ("#2563eb", "#059669", "#dc2626"), strict=False
        )
    }

    def series(ax, metric, scale=1.0):
        for prompt, color in colors.items():
            for implementation in IMPLEMENTATIONS:
                rows = sorted(
                    [
                        row
                        for row in summary
                        if row["prompt_length"] == prompt and row["implementation"] == implementation
                    ],
                    key=lambda row: row["generated_tokens"],
                )
                naive = implementation == IMPLEMENTATIONS[0]
                ax.errorbar(
                    [row["generated_tokens"] for row in rows],
                    [row[metric + "_mean"] / scale for row in rows],
                    yerr=[row[metric + "_stdev"] / scale for row in rows],
                    color=color,
                    linestyle="--" if naive else "-",
                    marker="o",
                    capsize=3,
                    label=f"P={prompt} {'naive' if naive else 'cached'}",
                )
        ax.set_xlabel("Generated tokens")
        ax.set_xticks(sorted({row["generated_tokens"] for row in summary}))
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)

    def save(fig, name):
        fig.savefig(output / (name + ".svg"), metadata={"Date": None})
        fig.savefig(output / (name + ".png"), dpi=150)
        plt.close(fig)

    for name, metric, ylabel, title in (
        ("decode-throughput", "decode_tokens_per_second", "Decode tokens/s (G−1 tokens)", "Decode throughput"),
        ("latency", "end_to_end_seconds", "End-to-end latency (s)", "Uninstrumented generation latency"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
        series(ax, metric)
        ax.set_ylabel(ylabel)
        ax.set_title(title + " · FP32 reference · batch 1 · mean ± sample std")
        ax.legend(ncol=2, fontsize=9)
        save(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(
        axes, ("peak_allocated_bytes", "peak_reserved_bytes"), ("Peak allocated", "Peak reserved"), strict=True
    ):
        series(ax, metric, 2**20)
        ax.set_title(title)
        ax.set_ylabel("MiB")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Peak GPU memory · plain API after warmup · mean ± sample std")
    save(fig, "peak-memory")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(
        axes, ("decode_speedup", "end_to_end_speedup"), ("Decode speedup", "End-to-end speedup"), strict=True
    ):
        for prompt, color in colors.items():
            rows = sorted(
                [row for row in derived["comparison"] if row["prompt_length"] == prompt and row["readme_eligible"]],
                key=lambda row: row["generated_tokens"],
            )
            ax.errorbar(
                [row["generated_tokens"] for row in rows],
                [row[metric + "_mean"] for row in rows],
                yerr=[row[metric + "_stdev"] for row in rows],
                marker="o",
                capsize=3,
                color=color,
                label=f"P={prompt}",
            )
        ax.axhline(1, color="#6b7280", linestyle=":", linewidth=1)
        ax.set_xlabel("Generated tokens")
        ax.set_xticks(sorted({row["generated_tokens"] for row in summary}))
        ax.set_ylabel("Naive / cached (×)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Paired run speedup · mean ± sample std · three independent processes per implementation")
    save(fig, "speedup")


def publish(source, destination):
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    derived = aggregate(report["results"], required_runs=report.get("independent_runs", 3))
    if not derived["comparison"] or not all(row["readme_eligible"] for row in derived["comparison"]):
        raise ValueError(
            "publication requires three clean, complete, parity-eligible independent runs per implementation"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Public projections retain every metric, hash, status and provenance, but
    # omit host-specific commands/stderr. Ignored source JSON remains unchanged.
    sanitized = {key: value for key, value in report.items() if key not in ("command",)}
    sanitized["results"] = [
        {key: value for key, value in row.items() if key not in ("worker_command", "worker_stderr")}
        for row in report["results"]
    ]
    sanitized["reproduction"] = "python scripts/benchmark_inference.py --output-dir artifacts/phase4-p4-02/formal"
    derived = write_results(sanitized, destination)
    rendered = tables(derived)
    (destination / "tables.md").write_text(
        "# Phase 4 inference benchmark\n\nMean ± sample standard deviation across three independent runs.\n\n"
        + "\n\n".join(rendered.values())
        + "\n",
        encoding="utf-8",
    )
    charts(derived, destination)
    return derived
