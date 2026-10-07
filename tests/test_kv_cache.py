"""Inference-only KV cache correctness and compatibility contracts."""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from lmforge.config import ModelConfig
from lmforge.nn.transformer import TransformerLM
from lmforge.tokenization.tokenizer import BPE_tokenizer
from lmforge.training import generate as generation
from lmforge.training.checkpoint import create_training_checkpoint, save_training_checkpoint


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(backend="reference", device="cpu"):
    torch.manual_seed(17)
    return TransformerLM(
        vocab_size=32,
        context_length=16,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10000,
        attention_backend=backend,
        device=device,
    ).eval()


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.inference_mode()
def test_full_logits_match_token_by_token_cache(backend, batch_size):
    lm = model(backend)
    ids = torch.randint(0, 32, (batch_size, 10))
    full = lm(ids)
    cache = lm.allocate_kv_cache(batch_size=batch_size)
    outputs = []
    for index in range(ids.shape[1]):
        outputs.append(lm(ids[:, index : index + 1], kv_cache=cache))
        assert cache.current_length == index + 1
        assert all(layer.current_length == index + 1 for layer in cache.layers)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@torch.inference_mode()
def test_prefill_decode_positions_projections_and_storage(backend):
    lm = model(backend)
    ids = torch.randint(0, 32, (2, 10))
    full = lm(ids)
    cache = lm.allocate_kv_cache(batch_size=2, max_seq_len=12)
    pointers = [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in cache.layers]
    seen = [[] for _ in lm.layers]
    positions = [[] for _ in lm.layers]
    handles = []
    for index, block in enumerate(lm.layers):
        for projection in (block.attn.k_proj, block.attn.v_proj):
            handles.append(
                projection.register_forward_pre_hook(
                    lambda module, args, index=index: seen[index].append(args[0].shape[-2])
                )
            )
        handles.append(
            block.attn.rope.register_forward_pre_hook(
                lambda module, args, index=index: positions[index].append(args[1].clone())
            )
        )
    try:
        prefill = lm(ids[:, :4], kv_cache=cache)
        torch.testing.assert_close(prefill, full[:, :4], rtol=3e-5, atol=3e-6)
        for index in range(4, 10):
            logits = lm(ids[:, index : index + 1], kv_cache=cache)
            torch.testing.assert_close(logits[:, -1], full[:, index], rtol=3e-5, atol=3e-6)
        for index, layer in enumerate(cache.layers):
            assert seen[index] == [4, 4] + [1, 1] * 6
            assert pointers[index] == (layer.key.data_ptr(), layer.value.data_ptr())
            assert layer.key.shape == (2, 4, 12, 4)
            assert layer.current_length == 10
            assert layer.batch_size == 2 and layer.max_seq_len == 12
            assert layer.device == ids.device and layer.dtype == torch.float32
            assert positions[index][2][..., 0].eq(4).all()
            assert positions[index][-1][..., 0].eq(9).all()
        cache.reset()
        assert cache.current_length == 0
        assert all(layer.current_length == 0 for layer in cache.layers)
        replacement = torch.tensor([[3, 2], [1, 4]])
        torch.testing.assert_close(lm(replacement, kv_cache=cache), lm(replacement))
        assert pointers == [(layer.key.data_ptr(), layer.value.data_ptr()) for layer in cache.layers]
    finally:
        for handle in handles:
            handle.remove()


def test_cache_requires_inference_mode_and_eval():
    lm = model()
    with torch.inference_mode():
        cache = lm.allocate_kv_cache(batch_size=1)
    ids = torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="inference_mode"):
        lm(ids, kv_cache=cache)
    with torch.no_grad(), pytest.raises(ValueError, match="inference_mode"):
        lm(ids, kv_cache=cache)
    with torch.inference_mode():
        lm.train()
        with pytest.raises(ValueError, match="eval"):
            lm(ids, kv_cache=cache)
    assert cache.current_length == 0


@torch.inference_mode()
def test_cache_validation_and_overflow_do_not_advance_lengths():
    lm = model()
    cache = lm.allocate_kv_cache(batch_size=2, max_seq_len=4)
    with pytest.raises(ValueError, match="batch"):
        lm(torch.tensor([[1]]), kv_cache=cache)
    with pytest.raises(ValueError, match="token_positions"):
        lm(torch.ones(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long), kv_cache=cache)
    assert cache.current_length == 0
    lm(torch.ones(2, 3, dtype=torch.long), kv_cache=cache)
    with pytest.raises(ValueError, match="single"):
        lm(torch.ones(2, 2, dtype=torch.long), kv_cache=cache)
    lm(torch.ones(2, 1, dtype=torch.long), kv_cache=cache)
    with pytest.raises(ValueError, match="max_seq_len"):
        lm(torch.ones(2, 1, dtype=torch.long), kv_cache=cache)
    assert all(layer.current_length == 4 for layer in cache.layers)
    cache.layers[0].current_length = 3
    with pytest.raises(ValueError, match="length"):
        lm(torch.ones(2, 1, dtype=torch.long), kv_cache=cache)
    cache.reset()
    with pytest.raises(ValueError, match="dtype"):
        lm.double()(torch.ones(2, 1, dtype=torch.long), kv_cache=cache)
    assert cache.current_length == 0


@pytest.mark.parametrize("batch_size,max_seq_len", [(0, 4), (1, 0), (1, 17), (True, 4)])
@torch.inference_mode()
def test_invalid_cache_allocation(batch_size, max_seq_len):
    with pytest.raises(ValueError):
        model().allocate_kv_cache(batch_size=batch_size, max_seq_len=max_seq_len)


@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@pytest.mark.parametrize("batched", [False, True])
def test_greedy_generation_parity_and_only_new_tokens(backend, batched):
    lm = model(backend).train()
    ids = torch.tensor([[1, 4, 7], [2, 3, 8]]) if batched else torch.tensor([1, 4, 7])
    original = ids.clone()
    naive = generation.generate_naive(lm, ids, max_new_tokens=6, temperature=0)
    lengths = []
    handle = lm.token_embeddings.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[-1]))
    try:
        cached = generation.generate_with_kv_cache(lm, ids, max_new_tokens=6, temperature=0)
    finally:
        handle.remove()
    assert torch.equal(cached, naive)
    assert lengths == [3, 1, 1, 1, 1, 1]
    assert torch.equal(ids, original)
    assert lm.training
    assert torch.equal(generation.generate(lm, ids, max_new_tokens=6, temperature=0), naive)


@pytest.mark.parametrize("function_name", ["generate_naive", "generate_with_kv_cache"])
def test_eos_and_zero_tokens(function_name):
    lm = model()
    function = getattr(generation, function_name)
    ids = torch.tensor([1, 2])
    assert torch.equal(function(lm, ids, max_new_tokens=0), ids)
    with torch.inference_mode():
        first = lm(ids[None])[0, -1].argmax().item()
    output = function(lm, ids, max_new_tokens=5, temperature=0, eos_token_id=first)
    assert output.tolist() == [1, 2, first]
    batch = torch.tensor([[1, 2], [3, 4]])
    separate = [function(lm, row, max_new_tokens=5, temperature=0, eos_token_id=first) for row in batch]
    together = function(lm, batch, max_new_tokens=5, temperature=0, eos_token_id=first)
    for index, row in enumerate(separate):
        assert torch.equal(together[index, : len(row)], row)
        assert together[index, len(row) :].eq(first).all()


def test_generation_capacity_and_mode_restoration():
    lm = model().train()
    with pytest.raises(ValueError, match="max_seq_len"):
        generation.generate_with_kv_cache(lm, [1, 2], max_new_tokens=15)
    with pytest.raises(ValueError, match="max_seq_len"):
        generation.generate_with_kv_cache(lm, [1] * 17, max_new_tokens=0)
    assert lm.training
    assert generation.generate_with_kv_cache(lm, [1, 2], max_new_tokens=2, context_length=4, temperature=0).shape == (
        4,
    )
    assert lm.training


@torch.inference_mode()
def test_reference_and_sdpa_cached_parity():
    reference, sdpa = model(), model("sdpa")
    sdpa.load_state_dict(reference.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    ref_cache, sdpa_cache = (lm.allocate_kv_cache(batch_size=1) for lm in (reference, sdpa))
    for chunk in (ids[:, :3], ids[:, 3:4], ids[:, 4:]):
        torch.testing.assert_close(
            reference(chunk, kv_cache=ref_cache), sdpa(chunk, kv_cache=sdpa_cache), rtol=3e-5, atol=3e-6
        )
    assert torch.equal(
        generation.generate_naive(reference, ids, max_new_tokens=5, temperature=0),
        generation.generate_with_kv_cache(sdpa, ids, max_new_tokens=5, temperature=0),
    )


def test_cache_does_not_change_state_dict_or_training_gradients():
    lm = model()
    keys = tuple(lm.state_dict())
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        cache = lm.allocate_kv_cache(batch_size=1)
        lm(ids, kv_cache=cache)
    assert tuple(lm.state_dict()) == keys
    assert all("cache" not in name for name in keys)
    lm.train()
    lm(ids).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in lm.parameters())


def test_checkpoint_text_and_installed_cli(tmp_path):
    config = ModelConfig(vocab_size=257, context_length=16, d_model=16, num_layers=2, num_heads=4, d_ff=32)
    torch.manual_seed(17)
    lm = TransformerLM(**asdict(config))
    tokenizer = BPE_tokenizer({i: bytes([i]) for i in range(256)}, [], ["<|endoftext|>"])
    state = create_training_checkpoint(
        iteration=0,
        config={"model": asdict(config)},
        model=lm.state_dict(),
        optimizer={},
        scaler={},
        best_val_loss=0.0,
        data_sha256={},
        tokenizer={"vocab": tokenizer.vocab, "merges": tokenizer.merges, "special_tokens": tokenizer.special_tokens},
        train_rng={},
        val_rng={},
        torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),
        cuda_rng=None,
    )
    path = tmp_path / "model.pt"
    save_training_checkpoint(state, path)
    loaded, tok = generation.load_model(path)
    expected = generation.generate_text(loaded, tok, "Hi", max_new_tokens=3, temperature=0)
    actual = generation.generate_text(loaded, tok, "Hi", max_new_tokens=3, temperature=0, use_kv_cache=True)
    assert actual == expected
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lmforge.training.generate",
            "--checkpoint",
            str(path),
            "--prompt",
            "Hi",
            "--max-new-tokens",
            "3",
            "--temperature",
            "0",
            "--kv-cache",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip("\n") == actual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["reference", "sdpa"])
@torch.inference_mode()
def test_cuda_fp32_cache_parity(backend):
    lm = model(backend, "cuda")
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    full = lm(ids)
    cache = lm.allocate_kv_cache(batch_size=1)
    parts = [lm(ids[:, :3], kv_cache=cache)]
    for index in (3, 4):
        parts.append(lm(ids[:, index : index + 1], kv_cache=cache))
    torch.testing.assert_close(torch.cat(parts, 1), full, rtol=5e-5, atol=5e-6)
    assert torch.equal(
        generation.generate_naive(lm, ids, max_new_tokens=5, temperature=0),
        generation.generate_with_kv_cache(lm, ids, max_new_tokens=5, temperature=0),
    )
