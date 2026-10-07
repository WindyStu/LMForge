"""Autoregressive decoding from a training checkpoint (temperature / top-p)."""
from __future__ import annotations

import math

import torch

from ..nn.transformer import TransformerLM
from ..tokenization.tokenizer import BPE_tokenizer
from . import checkpoint as checkpoint_io


def sampling_probs(logits, *, temperature=1.0, top_p=1.0):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and >= 0')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    logits = logits.float()
    if logits.shape[-1] == 0 or not torch.isfinite(logits).all():
        raise ValueError('logits must be nonempty and finite')
    if temperature == 0:
        return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    probabilities = torch.softmax((logits - logits.max(-1, keepdim=True).values) / temperature, dim=-1)
    if top_p < 1:
        ordered, indices = probabilities.sort(dim=-1, descending=True)
        # Keep the smallest prefix whose cumulative probability reaches top_p.
        remove = ordered.cumsum(-1) - ordered >= top_p
        ordered = ordered.masked_fill(remove, 0)
        ordered = ordered / ordered.sum(-1, keepdim=True)
        probabilities = torch.zeros_like(ordered).scatter(-1, indices, ordered)
    return probabilities


@torch.inference_mode()
def _generate(model, input_ids, *, max_new_tokens, temperature, top_p,
              eos_token_id, context_length, generator, use_kv_cache):
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if (not math.isfinite(temperature) or temperature < 0 or
            not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("invalid temperature or top_p")
    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, device=device)
    if (ids.ndim not in (1, 2) or ids.numel() == 0 or
            ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("input_ids must be a nonempty 1D or 2D integer sequence")
    unbatched = ids.ndim == 1
    ids = ids.long().reshape(1, -1) if unbatched else ids.long()
    limit = context_length if context_length is not None else model.context_length
    if type(limit) is not int or not 1 <= limit <= model.context_length:
        raise ValueError("context_length must be within the model context window")
    batch_size, prompt_length = ids.shape
    if use_kv_cache and prompt_length + max_new_tokens > limit:
        raise ValueError("prompt length + max_new_tokens exceeds KV cache max_seq_len")
    if max_new_tokens == 0:
        return ids[0].clone() if unbatched else ids.clone()
    output = torch.empty((batch_size, prompt_length + max_new_tokens),
                         device=device, dtype=torch.long)
    output[:, :prompt_length].copy_(ids)
    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
    length = prompt_length
    was_training = model.training
    cache = None
    model.eval()
    try:
        if use_kv_cache:
            cache = model.allocate_kv_cache(batch_size=batch_size, max_seq_len=limit)
        for step in range(max_new_tokens):
            if use_kv_cache:
                inputs = output[:, :length] if step == 0 else output[:, length - 1:length]
                logits = model(inputs, kv_cache=cache)[:, -1]
            else:
                logits = model(output[:, max(0, length - limit):length])[:, -1]
            probs = sampling_probs(logits, temperature=temperature, top_p=top_p)
            next_ids = (probs.argmax(-1) if temperature == 0 else
                        torch.multinomial(probs, 1, generator=generator).squeeze(-1))
            if eos_token_id is not None:
                next_ids = torch.where(finished, eos_token_id, next_ids)
                finished |= next_ids == eos_token_id
            output[:, length].copy_(next_ids)
            length += 1
            if eos_token_id is not None and finished.all():
                break
    finally:
        if cache is not None:
            cache.reset()
        model.train(was_training)
    # Do not retain unused output capacity after an early EOS.
    return output[0, :length].clone() if unbatched else output[:, :length].clone()


def generate_naive(model, input_ids, *, max_new_tokens=128, temperature=1.0, top_p=1.0,
                   eos_token_id=None, context_length=None, generator=None):
    """Recompute the newest window, resetting positions to zero each step.

    Return IDs including the prompt and terminal EOS. Accept a 1D sequence or
    a 2D equal-length batch; completed batch rows are padded with EOS.
    """
    return _generate(model, input_ids, max_new_tokens=max_new_tokens,
                     temperature=temperature, top_p=top_p, eos_token_id=eos_token_id,
                     context_length=context_length, generator=generator, use_kv_cache=False)


def generate_with_kv_cache(model, input_ids, *, max_new_tokens=128, temperature=1.0, top_p=1.0,
                           eos_token_id=None, context_length=None, generator=None):
    """Prefill once, then process one new token per step using request-owned K/V.

    The prompt plus requested output must fit context_length (default: model
    context). No cropping or position reset is performed. EOS/batch return
    semantics match generate_naive.
    """
    return _generate(model, input_ids, max_new_tokens=max_new_tokens,
                     temperature=temperature, top_p=top_p, eos_token_id=eos_token_id,
                     context_length=context_length, generator=generator, use_kv_cache=True)


def generate(model, input_ids, *, max_new_tokens=128, temperature=1.0, top_p=1.0,
             eos_token_id=None, context_length=None, generator=None, use_kv_cache=False):
    """Compatible generation entry point; naive is the explicit default."""
    implementation = generate_with_kv_cache if use_kv_cache else generate_naive
    return implementation(model, input_ids, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p, eos_token_id=eos_token_id,
                          context_length=context_length, generator=generator)


def load_model(checkpoint_path, *, device='cpu'):
    checkpoint = checkpoint_io.load_training_checkpoint(checkpoint_path)
    model = TransformerLM(**checkpoint['config']['model'], device=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    saved_tokenizer = checkpoint.get('tokenizer')
    tokenizer = BPE_tokenizer(**saved_tokenizer) if saved_tokenizer is not None else None
    return model, tokenizer


def generate_text(model, tokenizer, prompt, *, eos_token='<|endoftext|>', seed=0, **kwargs):
    if tokenizer is None:
        raise ValueError('text generation requires a tokenizer saved in the checkpoint')
    eos_id = (tokenizer.reverse_vocab.get(eos_token.encode('utf-8'))
              if eos_token is not None and eos_token in tokenizer.special_tokens else None)
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        if eos_id is None:
            raise ValueError('empty prompt needs an EOS special token as its initial token')
        prompt_ids = [eos_id]
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    output = generate(model, torch.tensor(prompt_ids, device=device),
                      eos_token_id=eos_id, generator=rng, **kwargs)
    continuation = output[len(prompt_ids):].tolist()
    if continuation and continuation[-1] == eos_id:
        continuation.pop()
    return prompt + tokenizer.decode(continuation)


def main():
    from ..cli import legacy_main

    return legacy_main("generate")


if __name__ == '__main__':
    raise SystemExit(main())
