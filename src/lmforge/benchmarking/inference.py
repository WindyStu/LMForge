"""Isolated inference API measurements, with first-token boundary hooks only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..nn.transformer import TransformerLM
from ..training import generate as generation
from .environment import benchmark_metadata
from .inference_results import IMPLEMENTATIONS, aggregate, write_results
from .runner import is_oom

MODEL = {"vocab_size": 8192, "d_model": 256, "num_layers": 4, "num_heads": 4, "d_ff": 768, "rope_theta": 10000.0}


def cache_bytes(*, batch_size, num_layers, max_seq_len, d_model, element_size):
    return 2 * batch_size * num_layers * max_seq_len * d_model * element_size


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def probe_cache(model, *, batch_size):
    device = next(model.parameters()).device
    _sync(device)
    before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    cache = model.allocate_kv_cache(batch_size=batch_size)
    _sync(device)
    after = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    storage = sum(
        layer.key.numel() * layer.key.element_size() + layer.value.numel() * layer.value.element_size()
        for layer in cache.layers
    )
    theoretical = cache_bytes(
        batch_size=batch_size,
        num_layers=model.num_layers,
        max_seq_len=model.context_length,
        d_model=model.layers[0].attn.d_model,
        element_size=next(model.parameters()).element_size(),
    )
    del cache
    return {
        "cache_allocated_bytes": storage,
        "cache_theoretical_bytes": theoretical,
        "cache_cuda_allocated_delta_bytes": after - before,
    }


def measure_generation(model, ids, *, implementation, generated_tokens, device, phases):
    if generated_tokens < 2:
        raise ValueError("phase benchmark requires at least two generated tokens")
    function = getattr(generation, implementation)
    handles = []
    boundaries = {}
    calls = 0

    def before_forward(_module, _args):
        nonlocal calls
        calls += 1
        if calls == 1:
            _sync(device)
            boundaries["prefill_start"] = time.perf_counter()
        elif calls == 2:
            # Token 1 has been sampled and committed before this forward begins.
            _sync(device)
            boundaries["first_ready"] = time.perf_counter()
            for handle in handles:
                handle.remove()

    def after_forward(_module, _args, _output):
        if calls == 1:
            _sync(device)
            boundaries["prefill_end"] = time.perf_counter()

    if phases:
        handles = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward)]
    try:
        _sync(device)
        start = time.perf_counter()
        output = function(model, ids, max_new_tokens=generated_tokens, temperature=0, eos_token_id=None)
        _sync(device)
        end = time.perf_counter()
    finally:
        for handle in handles:
            handle.remove()
    if not phases:
        return {"end_to_end_seconds": end - start}, output
    if "first_ready" not in boundaries:
        raise ValueError("generation did not reach the decode boundary")
    decode_seconds = end - boundaries["first_ready"]
    return {
        "prefill_seconds": boundaries["prefill_end"] - boundaries["prefill_start"],
        "ttft_seconds": boundaries["first_ready"] - start,
        "decode_seconds": decode_seconds,
        "decode_tokens": generated_tokens - 1,
        "decode_tokens_per_second": (generated_tokens - 1) / decode_seconds,
        "phase_total_seconds": end - start,
    }, output


def matrix(prompts, generated, *, runs):
    for value in (*prompts, *generated):
        if value < 1:
            raise ValueError("lengths must be positive")
    slots = []
    for run in range(1, runs + 1):
        order = IMPLEMENTATIONS if run % 2 else IMPLEMENTATIONS[::-1]
        for prompt in prompts:
            for tokens in generated:
                slots.extend((run, prompt, tokens, impl) for impl in order)
    return slots


def _configuration(args, prompt, tokens, implementation):
    return {
        "prompt_length": prompt,
        "generated_tokens": tokens,
        "implementation": implementation,
        "batch_size": 1,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "warmup": args.warmup,
        "attention_backend": args.attention_backend,
        "precision": "float32",
        "device": args.device,
        "execution_mode": "eager",
        "weight_source": "seeded_random",
        "model": MODEL,
    }


def _workload(args, config):
    if config["prompt_length"] + config["generated_tokens"] > args.max_seq_len:
        raise ValueError("prompt + output exceeds benchmark max_seq_len")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # CPU construction fixes identical FP32 weight bytes in every worker.
    model = TransformerLM(**MODEL, context_length=args.max_seq_len, attention_backend=args.attention_backend).eval()
    weight_digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        weight_digest.update(name.encode("utf-8"))
        weight_digest.update(parameter.detach().contiguous().numpy().tobytes())
    weights_sha256 = weight_digest.hexdigest()
    model.to(args.device)
    ids_cpu = torch.randint(
        0, MODEL["vocab_size"], (config["prompt_length"],), generator=torch.Generator().manual_seed(args.seed + 1)
    )
    input_sha256 = _tensor_hash(ids_cpu)
    ids = ids_cpu.to(args.device)
    cache_metrics = dict.fromkeys(
        ("cache_allocated_bytes", "cache_theoretical_bytes", "cache_cuda_allocated_delta_bytes"), 0
    )
    if config["implementation"] == "generate_with_kv_cache":
        cache_metrics = probe_cache(model, batch_size=1)
        if cache_metrics["cache_allocated_bytes"] != cache_metrics["cache_theoretical_bytes"]:
            raise ValueError("theoretical cache bytes do not match storage")
    function = getattr(generation, config["implementation"])
    for _ in range(args.warmup):
        warm = function(model, ids, max_new_tokens=config["generated_tokens"], temperature=0, eos_token_id=None)
        del warm
    gc.collect()
    _sync(args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    plain, output = measure_generation(
        model,
        ids,
        implementation=config["implementation"],
        generated_tokens=config["generated_tokens"],
        device=args.device,
        phases=False,
    )
    peak = {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    if torch.device(args.device).type == "cuda":
        peak = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        }
    expected_hash = _tensor_hash(output)
    del output
    phased, output = measure_generation(
        model,
        ids,
        implementation=config["implementation"],
        generated_tokens=config["generated_tokens"],
        device=args.device,
        phases=True,
    )
    output_hash = _tensor_hash(output)
    if expected_hash != output_hash:
        raise ValueError("instrumented/plain API greedy outputs differ")
    return {
        "input_sha256": input_sha256,
        "weights_sha256": weights_sha256,
        "output_sha256": output_hash,
        "internal_parity": True,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "measurement": {**plain, **phased, **peak, **cache_metrics},
    }


def worker(args):
    config = _configuration(args, args.prompt, args.tokens, args.implementation)
    metadata = benchmark_metadata("inference", seed=args.seed)
    record = {
        "configuration": config,
        "run": args.run,
        "worker_pid": os.getpid(),
        "git": metadata["git"],
        "environment": metadata["environment"],
    }
    try:
        record.update(_workload(args, config))
        record["status"] = "ok"
        if (
            args.device.startswith("cuda")
            and record["measurement"]["peak_reserved_bytes"]
            > torch.cuda.get_device_properties(args.device).total_memory
        ):
            record["status"] = "memory_capacity_exceeded"
            record["failure"] = {
                "type": "DeviceMemoryCapacityExceeded",
                "message": "WSL CUDA peak reserved memory exceeded physical GPU capacity",
            }
    except Exception as error:  # noqa: BLE001 - preserve every benchmark failure
        record["status"] = "oom" if is_oom(error) else "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        if args.device.startswith("cuda") and torch.cuda.is_initialized():
            record["failure"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            record["failure"]["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
    print(json.dumps(record), flush=True)
    return 0


def _csv_ints(value):
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 1 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("provide distinct positive comma-separated lengths")
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=Path("artifacts/phase4-p4-02/formal"))
    result.add_argument("--prompt-lengths", type=_csv_ints, default=[128, 512, 1024])
    result.add_argument("--generated-tokens", type=_csv_ints, default=[64, 128, 512])
    result.add_argument("--runs", type=int, default=3)
    result.add_argument("--warmup", type=int, default=3)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--max-seq-len", type=int, default=2048)
    result.add_argument("--device", default="cuda")
    result.add_argument("--attention-backend", choices=("reference", "sdpa"), default="reference")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--prompt", type=int, help=argparse.SUPPRESS)
    result.add_argument("--tokens", type=int, help=argparse.SUPPRESS)
    result.add_argument("--implementation", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    result.add_argument("--run", type=int, help=argparse.SUPPRESS)
    return result


def main():
    args = parser().parse_args()
    if args.worker:
        return worker(args)
    if args.runs < 3 or args.warmup < 1:
        raise SystemExit("formal benchmark requires >=3 independent runs and >=1 warmup")
    if min(args.generated_tokens) < 2 or max(args.prompt_lengths) + max(args.generated_tokens) > args.max_seq_len:
        raise SystemExit("generated tokens must be >=2 and prompt + output must fit max_seq_len")
    if (args.output_dir / "benchmark.json").exists():
        raise SystemExit("output already exists; choose a fresh directory to preserve previous attempts")
    report = benchmark_metadata("inference", seed=args.seed)
    report.update(
        {
            "format": "lmforge-inference-benchmark-v1",
            "independent_runs": args.runs,
            "protocol": {
                "warmup_full_generations": args.warmup,
                "phase_measurement": "first/second forward boundary hooks removed before decode",
                "end_to_end_measurement": "separate synchronized API call without hooks",
                "decode_tokens": "generated_tokens - 1",
                "memory": "peak allocated/reserved in plain API call after warmup; reserved includes warmup allocator pools",
                "weights": "same seeded random FP32 weights; context=2048 is untrained performance fixture",
                "input": "CPU torch.randint, independent generator seed+1; no tokenization",
                "tf32": False,
                "compile": False,
                "profiler": False,
            },
            "inputs": {
                str(length): torch.randint(
                    0, MODEL["vocab_size"], (length,), generator=torch.Generator().manual_seed(args.seed + 1)
                ).tolist()
                for length in args.prompt_lengths
            },
            "results": [],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run, prompt, tokens, implementation in matrix(args.prompt_lengths, args.generated_tokens, runs=args.runs):
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / "scripts" / "benchmark_inference.py"),
            "--worker",
            "--run",
            str(run),
            "--prompt",
            str(prompt),
            "--tokens",
            str(tokens),
            "--implementation",
            implementation,
            "--warmup",
            str(args.warmup),
            "--max-seq-len",
            str(args.max_seq_len),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--attention-backend",
            args.attention_backend,
        ]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
        child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False, env=env)
        try:
            record = json.loads(child.stdout.strip().splitlines()[-1])
            if child.returncode:
                raise ValueError(f"child return code {child.returncode}")
        except (ValueError, IndexError) as error:
            record = {
                "configuration": _configuration(args, prompt, tokens, implementation),
                "run": run,
                "status": "failed",
                "failure": {
                    "type": "WorkerProcessError",
                    "message": str(error),
                    "stdout": child.stdout,
                    "stderr": child.stderr,
                },
            }
        record["worker_command"] = command
        record["worker_stderr"] = child.stderr
        report["results"].append(record)
        (args.output_dir / f"run-{run}-p{prompt}-g{tokens}-{implementation}.json").write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )
        write_results(report, args.output_dir)
        print(
            f"{len(report['results'])}/{2 * args.runs * len(args.prompt_lengths) * len(args.generated_tokens)} "
            f"run={run} p={prompt} g={tokens} {implementation}: {record['status']}",
            flush=True,
        )
    comparisons = aggregate(report["results"], required_runs=args.runs)["comparison"]
    return 0 if all(row["readme_eligible"] for row in comparisons) else 1
