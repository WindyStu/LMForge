"""Inference benchmark timing, provenance and publication gates."""

from __future__ import annotations

import copy
import csv
import json

import pytest
import torch


def _benchmark():
    from lmforge.benchmarking import inference

    return inference


def _small_model():
    from lmforge.nn.transformer import TransformerLM

    torch.manual_seed(42)
    return TransformerLM(
        vocab_size=32, context_length=16, d_model=16, num_layers=2, num_heads=4, d_ff=32, rope_theta=10000
    ).eval()


def test_theoretical_cache_memory():
    benchmark = _benchmark()
    assert (
        benchmark.cache_bytes(batch_size=1, num_layers=4, max_seq_len=2048, d_model=256, element_size=4) == 16 * 2**20
    )
    assert (
        benchmark.cache_bytes(batch_size=2, num_layers=4, max_seq_len=1024, d_model=256, element_size=4) == 16 * 2**20
    )


@pytest.mark.parametrize("implementation", ["generate_naive", "generate_with_kv_cache"])
def test_phase_timing_calls_real_api_and_removes_hooks(implementation):
    from lmforge.training import generate

    benchmark = _benchmark()
    torch.set_num_threads(1)
    lm = _small_model()
    ids = torch.tensor([1, 2, 3])
    expected = getattr(generate, implementation)(lm, ids, max_new_tokens=5, temperature=0)
    result, output = benchmark.measure_generation(
        lm, ids, implementation=implementation, generated_tokens=5, device="cpu", phases=True
    )
    assert torch.equal(output, expected)
    assert result["decode_tokens"] == 4
    assert result["prefill_seconds"] > 0
    assert result["ttft_seconds"] >= result["prefill_seconds"]
    assert result["decode_seconds"] > 0
    assert result["phase_total_seconds"] == pytest.approx(result["ttft_seconds"] + result["decode_seconds"])
    assert result["decode_tokens_per_second"] == pytest.approx(4 / result["decode_seconds"])
    assert len(lm._forward_hooks) == len(lm._forward_pre_hooks) == 0
    plain, plain_output = benchmark.measure_generation(
        lm, ids, implementation=implementation, generated_tokens=5, device="cpu", phases=False
    )
    assert torch.equal(plain_output, expected)
    assert plain["end_to_end_seconds"] > 0
    assert "ttft_seconds" not in plain


def test_phase_hooks_are_removed_after_failure():
    benchmark = _benchmark()
    lm = _small_model()
    with pytest.raises(ValueError):
        benchmark.measure_generation(
            lm,
            torch.tensor([1, 2]),
            implementation="generate_with_kv_cache",
            generated_tokens=20,
            device="cpu",
            phases=True,
        )
    assert len(lm._forward_hooks) == len(lm._forward_pre_hooks) == 0


def _records():
    rows = []
    for run in (1, 2, 3):
        for implementation in ("generate_naive", "generate_with_kv_cache"):
            naive = implementation == "generate_naive"
            rows.append(
                {
                    "run": run,
                    "worker_pid": run * 10 + naive,
                    "status": "ok",
                    "configuration": {
                        "prompt_length": 128,
                        "generated_tokens": 64,
                        "implementation": implementation,
                        "attention_backend": "reference",
                        "precision": "float32",
                        "device": "cuda",
                        "batch_size": 1,
                        "max_seq_len": 2048,
                        "seed": 42,
                        "warmup": 3,
                        "model": {
                            "vocab_size": 8192,
                            "d_model": 256,
                            "num_layers": 4,
                            "num_heads": 4,
                            "d_ff": 768,
                            "rope_theta": 10000,
                        },
                        "execution_mode": "eager",
                        "weight_source": "seeded_random",
                    },
                    "git": {"sha": "frozen", "dirty": False},
                    "environment": {"torch": "2.6", "gpus": [{"name": "test GPU", "total_memory_bytes": 2**30}]},
                    "input_sha256": "same-input",
                    "weights_sha256": "same-weights",
                    "output_sha256": "same-output",
                    "internal_parity": True,
                    "measurement": {
                        "ttft_seconds": run * 0.01,
                        "prefill_seconds": run * 0.008,
                        "decode_seconds": float(run) * (2 if naive else 1),
                        "decode_tokens_per_second": 63 / (run * (2 if naive else 1)),
                        "end_to_end_seconds": float(run) * (4 if naive else 2),
                        "peak_allocated_bytes": 100,
                        "peak_reserved_bytes": 200,
                        "cache_allocated_bytes": 0 if naive else 16 * 2**20,
                        "cache_cuda_allocated_delta_bytes": 0 if naive else 16 * 2**20,
                        "cache_theoretical_bytes": 0 if naive else 16 * 2**20,
                    },
                }
            )
    return rows


def test_three_run_aggregation_and_paired_speedups():
    benchmark = _benchmark()
    result = benchmark.aggregate(_records(), required_runs=3)
    assert len(result["summary"]) == 2
    assert all(row["readme_eligible"] for row in result["summary"])
    naive = result["summary"][0]
    assert naive["decode_seconds_mean"] == 4
    assert naive["decode_seconds_stdev"] == 2
    assert result["comparison"][0]["decode_speedup_mean"] == 2
    assert result["comparison"][0]["decode_speedup_stdev"] == 0
    assert result["comparison"][0]["end_to_end_speedup_mean"] == 2


@pytest.mark.parametrize(
    "mutation", ["missing", "oom", "dirty", "input", "weights", "backend", "output", "duplicate", "pid"]
)
def test_publication_rejects_incomplete_or_unfair_comparisons(mutation):
    benchmark = _benchmark()
    rows = copy.deepcopy(_records())
    if mutation == "missing":
        rows.pop()
    elif mutation == "oom":
        rows[-1]["status"] = "oom"
        rows[-1]["failure"] = {"type": "OutOfMemoryError", "message": "retained"}
    elif mutation == "dirty":
        rows[-1]["git"]["dirty"] = True
    elif mutation in ("input", "weights", "output"):
        rows[-1][mutation + "_sha256"] = "different"
    elif mutation == "backend":
        rows[-1]["configuration"]["attention_backend"] = "sdpa"
    elif mutation == "duplicate":
        rows[-1]["run"] = 2
    elif mutation == "pid":
        rows[-1]["worker_pid"] = rows[-2]["worker_pid"]
    result = benchmark.aggregate(rows, required_runs=3)
    assert not any(row["readme_eligible"] for row in result["comparison"])
    assert len(result["raw"]) == len(rows)
    if mutation == "oom":
        assert result["raw"][-1]["failure"]["message"] == "retained"


def test_report_retains_raw_json_csv_and_aggregates(tmp_path):
    benchmark = _benchmark()
    report = {"format": "lmforge-inference-benchmark-v1", "results": _records()}
    benchmark.write_results(report, tmp_path)
    assert json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8")) == report
    with (tmp_path / "raw-runs.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 6
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    with (tmp_path / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_matrix_has_54_unique_independent_slots():
    benchmark = _benchmark()
    slots = benchmark.matrix([128, 512, 1024], [64, 128, 512], runs=3)
    assert len(slots) == 54
    assert len(set(slots)) == 54
    assert slots[0][-1] == "generate_naive"
    assert slots[18][-1] == "generate_with_kv_cache"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cache_probe_matches_storage_and_allocator_delta():
    benchmark = _benchmark()
    lm = _small_model().cuda()
    result = benchmark.probe_cache(lm, batch_size=1)
    assert result["cache_allocated_bytes"] == 2 * 2 * 1 * 16 * 16 * 4
    assert result["cache_cuda_allocated_delta_bytes"] == result["cache_allocated_bytes"]
    assert result["cache_theoretical_bytes"] == result["cache_allocated_bytes"]


def test_publication_exports_four_figures_and_traceable_tables(tmp_path):
    from lmforge.benchmarking.inference_results import publish

    report = {"format": "lmforge-inference-benchmark-v1", "results": _records()}
    source = tmp_path / "raw"
    _benchmark().write_results(report, source)
    destination = tmp_path / "public"
    publish(source / "benchmark.json", destination)
    assert (destination / "benchmark.json").exists()
    assert (destination / "tables.md").read_text(encoding="utf-8").count("128") >= 1
    for name in ("decode-throughput.svg", "latency.svg", "peak-memory.svg", "speedup.svg"):
        assert "<svg" in (destination / name).read_text(encoding="utf-8")


def test_publication_rejects_missing_runs_and_keeps_original(tmp_path):
    from lmforge.benchmarking.inference_results import publish

    rows = _records()[:-1]
    source = tmp_path / "raw"
    _benchmark().write_results({"results": rows}, source)
    with pytest.raises(ValueError, match="three|eligible"):
        publish(source / "benchmark.json", tmp_path / "public")
    assert len(json.loads((source / "benchmark.json").read_text(encoding="utf-8"))["results"]) == 5
