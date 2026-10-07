# Inference and KV cache

Generation is available from `lmforge.training.generate`:

- `generate_naive(model, input_ids, ...)` recomputes the most recent context
  window, with positions starting at zero each step.
- `generate_with_kv_cache(model, input_ids, ...)` processes the full prompt once,
  then one new token per decoding step.
- `generate(..., use_kv_cache=False)` retains naive behavior by default.
- `generate_text(..., use_kv_cache=True)` selects cached checkpoint text generation.

Both token APIs accept a nonempty 1D integer sequence or a 2D equal-length batch
and return the prompt followed by generated IDs. A generated EOS is included.
Completed batch rows are padded with EOS until every row finishes. Greedy
generation uses `temperature=0`; temperature/top-p sampling remains available.
The caller's model training mode is restored on success or failure.

## Checkpoint CLI

```bash
uv run lmforge generate --checkpoint path/to/best.pt \
  --prompt "Once upon a time" --max-new-tokens 32 --temperature 0 --kv-cache
```

The flag defaults to off. Cached generation requires
`prompt_length + max_new_tokens <= context_length`, where the default limit
is the checkpoint model's context length. An over-capacity request fails before
generation, even if it could have stopped at EOS early. Cached generation never
crops history. Checkpoint schema, parameter names, and training config are unchanged.

## Direct cache API

```python
import torch

model.eval()
with torch.inference_mode():
    cache = model.allocate_kv_cache(batch_size=prompt_ids.shape[0])
    prompt_logits = model(prompt_ids, kv_cache=cache)  # [B, T, vocab]
    next_ids = prompt_logits[:, -1].argmax(-1, keepdim=True)
    next_logits = model(next_ids, kv_cache=cache)     # [B, 1, vocab]
    cache.reset()                                   # reuse allocated storage
```

An empty cache accepts a complete prompt (prefill). A nonempty cache accepts
exactly one new token per batch row. Every layer has independent `key`/`value`
storage in `[batch, heads, max_seq_len, head_dim]` layout and records
`current_length`, `max_seq_len`, `batch_size`, `device`, and `dtype`.
Keys are stored after RoPE. Positions are derived from cache length; supplying
explicit `token_positions` with a cache is rejected.

Storage is allocated once and updated by slice copies. Attention reads only
the valid prefix. Prefill is causal; single-token decode attends to every
stored position, including its own. Reference and SDPA backends are supported.
Caches are external request state and never appear in model state dictionaries.
The final sampled token is not processed again, so generation's internal cache
does not include that token.

Cache use requires eval mode and `torch.inference_mode()`. Allocation inherits
model projection device/dtype; changing the model device/dtype requires a new
cache. Autocast that changes projection dtype is rejected rather than silently
casting stored K/V. FP32 eager inference is the correctness reference; cached
compile and mixed-precision performance are not validated by this change.

`reset()` clears lengths without reallocating or zeroing storage; stale slots
cannot participate in attention. A computation failure after layer updates
resets every layer, so callers must prefill again. Generation owns and releases
its cache when it returns or raises. Direct API callers own cache lifetime and
can release storage by dropping the cache reference.

Ragged prompts, sliding windows, beam search, paged attention, continuous
batching, speculative decoding, and quantized caches are outside this API.
No inference throughput or memory speedup claim is made without a separate
benchmark.
