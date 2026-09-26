import json
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest
import torch

from lmforge.training.loss import clip_gradients


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_clip_mixed_shapes_and_generator():
    params = [torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(3))]
    for p in params:
        p.grad = torch.ones_like(p)
    clip_gradients(iter(params), 1.0)
    assert torch.sqrt(sum(p.grad.square().sum() for p in params)).item() == pytest.approx(1.0, abs=1e-6)
    clip_gradients(iter([]), 1.0)


class ScriptedLM(torch.nn.Module):
    context_length = 3

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, ids):
        self.seen.append(ids.clone())
        logits = torch.full((*ids.shape, 4), -100.0, device=ids.device)
        logits[..., 2 if len(self.seen) == 1 else 3] = 100.0
        assert not torch.is_grad_enabled()
        assert not self.training
        return logits


def test_generate_crops_recent_context_and_stops_at_eos():
    from lmforge.training.generate import generate

    model = ScriptedLM().train()
    prompt = torch.tensor([0, 1, 0, 1])
    output = generate(model, prompt, max_new_tokens=10, temperature=0, eos_token_id=3)
    assert output.tolist() == [0, 1, 0, 1, 2, 3]
    assert model.seen[0].tolist() == [[1, 0, 1]]
    assert model.seen[1].tolist() == [[0, 1, 2]]
    assert model.training
    assert prompt.tolist() == [0, 1, 0, 1]


def test_nucleus_keeps_the_token_crossing_threshold():
    from lmforge.training.generate import sampling_probs

    logits = torch.tensor([0.5, 0.3, 0.2]).log()
    assert torch.allclose(sampling_probs(logits, temperature=1, top_p=0.6), torch.tensor([0.625, 0.375, 0.0]))
    assert torch.equal(sampling_probs(logits, temperature=0, top_p=1), torch.tensor([1., 0., 0.]))
    with pytest.raises(ValueError):
        sampling_probs(logits, temperature=1, top_p=0)


def test_prepare_train_resume_and_generate(tmp_path):
    from lmforge.tokenization.tokenizer import BPE_tokenizer
    from lmforge.training.prepare import prepare_tokens
    from lmforge.training.train import ModelConfig, TrainConfig, train
    from lmforge.training.generate import load_model, generate_text

    tok = BPE_tokenizer({i: bytes([i]) for i in range(256)}, [], ['<|endoftext|>'])
    text_path = tmp_path / 'tiny.txt'
    text_path.write_text('hello world!\n' * 20, encoding='utf-8')
    tokens_path = tmp_path / 'tiny.npy'
    prepare_tokens(text_path, tokens_path, tok)
    tokens = np.load(tokens_path, mmap_mode='r')
    assert tok.decode(tokens.tolist()) == text_path.read_bytes().decode('utf-8')
    cfg = TrainConfig(model=ModelConfig(vocab_size=257, context_length=8, d_model=16,
        num_layers=1, num_heads=2, d_ff=32), max_steps=4, batch_size=2,
        grad_accum_steps=2, warmup_steps=0, lr_decay_steps=10, eval_interval=2,
        eval_batches=2, save_interval=2, log_interval=1, device='cpu')
    full = train(cfg, tokens, tokens, tmp_path / 'full', tokenizer=tok)
    train(replace(cfg, max_steps=2), tokens, tokens, tmp_path / 'resumed', tokenizer=tok)
    resumed = train(cfg, tokens, tokens, tmp_path / 'resumed',
        resume=tmp_path / 'resumed' / 'last.pt', tokenizer=tok)
    assert resumed['iteration'] == 4
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
    assert all(torch.isfinite(p).all() for p in resumed['model'].values())
    model, loaded_tok = load_model(tmp_path / 'resumed' / 'last.pt', device='cpu')
    output = generate_text(model, loaded_tok, 'hello', max_new_tokens=2, temperature=0)
    assert output.startswith('hello')
    logs = [
        json.loads(line)
        for line in (tmp_path / 'resumed' / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
    ]
    assert [r['step'] for r in logs] == [1, 2, 3, 4]
    assert (tmp_path / 'resumed' / 'best.pt').exists()


def test_attention_without_mask_and_device():
    from lmforge.nn.attention import scaled_dot_product_attention

    q, k, v = (torch.randn(2, 3, 4) for _ in range(3))
    expected = torch.softmax(q @ k.transpose(-2, -1) / 2, -1) @ v
    torch.testing.assert_close(scaled_dot_product_attention(q, k, v), expected)


def test_decode_utf8_joins_bytes_before_decoding():
    from lmforge.tokenization.tokenizer import BPE_tokenizer

    tok = BPE_tokenizer({i: bytes([i]) for i in range(256)}, [])
    assert tok.decode(list('你好🙂'.encode('utf-8'))) == '你好🙂'
    assert tok.decode([255]) == '\ufffd'
    assert tok.decode([]) == ''
    with pytest.raises(ValueError, match='unknown token ID'):
        tok.decode([256])


def test_training_rejects_invalid_data_and_config(tmp_path):
    from lmforge.training.train import ModelConfig, TrainConfig, train

    cfg = TrainConfig(model=ModelConfig(vocab_size=8, context_length=4), max_steps=1)
    with pytest.raises(ValueError, match='context_length'):
        train(cfg, np.array([1, 2]), None, tmp_path)
    with pytest.raises(ValueError, match='outside'):
        train(cfg, np.array([0, 1, 2, 3, 8]), None, tmp_path)
    with pytest.raises(ValueError, match='warmup'):
        replace(cfg, warmup_steps=1000).validate()


def test_accumulation_matches_one_larger_batch(tmp_path):
    from lmforge.training.train import ModelConfig, TrainConfig, train

    data = np.arange(80) % 8
    cfg = TrainConfig(model=ModelConfig(vocab_size=8, context_length=4, d_model=8,
        num_heads=2, num_layers=1, d_ff=16), batch_size=4, grad_accum_steps=1,
        max_steps=1, warmup_steps=0)
    large = train(cfg, data, None, tmp_path / 'large')
    small = train(replace(cfg, batch_size=2, grad_accum_steps=2), data, None, tmp_path / 'small')
    for key in large['model']:
        torch.testing.assert_close(large['model'][key], small['model'][key], atol=1e-6, rtol=1e-5)


def test_legacy_prepare_arguments_report_config_migration():
    result = subprocess.run(
        [sys.executable, '-m', 'lmforge.training.prepare', '--input', 'input.txt'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=30,
    )

    assert result.returncode == 2
    assert 'lmforge prepare' in result.stderr
    assert 'the following arguments are required: --config' in result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA device')
def test_cuda_fp16_training_and_decoding(tmp_path):
    from lmforge.training.train import ModelConfig, TrainConfig, train
    from lmforge.training.generate import load_model, generate

    cfg = TrainConfig(model=ModelConfig(vocab_size=8, context_length=4, d_model=8,
        num_heads=2, num_layers=1, d_ff=16), max_steps=2, batch_size=1,
        grad_accum_steps=1, warmup_steps=0, device='cuda', precision='float16')
    train(cfg, np.arange(80) % 8, None, tmp_path)
    model, _ = load_model(tmp_path / 'last.pt', device='cuda')
    assert generate(model, [1, 2], max_new_tokens=2).numel() == 4
