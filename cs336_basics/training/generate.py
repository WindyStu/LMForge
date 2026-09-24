"""Autoregressive decoding from a training checkpoint (temperature / top-p)."""
from __future__ import annotations

import argparse
import math

import torch

from cs336_basics.nn.transformer import TransformerLM
from cs336_basics.tokenization.tokenizer import BPE_tokenizer


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
def generate(model, input_ids, *, max_new_tokens=128, temperature=1.0, top_p=1.0,
             eos_token_id=None, context_length=None, generator=None):
    """Generate one sequence; return 1D IDs including prompt and terminal EOS.

    Recompute the newest context window each step, with positions reset to zero.
    This implementation does not use a KV cache.
    """
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError('max_new_tokens must be a non-negative integer')
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError('invalid temperature or top_p')
    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, device=device)
    if ids.ndim != 1 or ids.numel() == 0 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('input_ids must be a nonempty 1D integer sequence')
    ids = ids.long().clone()
    limit = context_length if context_length is not None else model.context_length
    if limit < 1 or limit > model.context_length:
        raise ValueError('context_length must be within the model context window')
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[-limit:].unsqueeze(0))[0, -1]
            probs = sampling_probs(logits, temperature=temperature, top_p=top_p)
            next_id = (probs.argmax().reshape(1) if temperature == 0 else
                       torch.multinomial(probs, 1, generator=generator))
            ids = torch.cat((ids, next_id))
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return ids


def load_model(checkpoint_path, *, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'cs336-training-v1':
        raise ValueError('expected a checkpoint written by training.train')
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--eos-token', default='<|endoftext|>')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    model, tokenizer = load_model(args.checkpoint, device=args.device)
    print(generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_p=args.top_p,
                        eos_token=args.eos_token, seed=args.seed))


if __name__ == '__main__':
    main()
