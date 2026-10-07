import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .ffn import SwiGLU
from .kv_cache import KVCache, LayerKVCache
from .linear import Embedding, Linear
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        device=None,
        dtype=None,
        attention_backend: str = "reference",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_heads
        self.d_ff = d_ff
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            True,
            max_seq_len,
            theta,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
        )
        self.ffn = SwiGLU(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_features,
        token_positions: torch.Tensor | None = None,
        *,
        kv_cache: LayerKVCache | None = None,
    ):
        """

        :param in_features:
        :return:
        """
        # 不使用 +=，避免修改调用者 Tensor 和 autograd 问题。
        in_features = in_features + self.attn(
            self.ln1(in_features),
            token_positions=token_positions,
            **({} if kv_cache is None else {"kv_cache": kv_cache}),
        )
        in_features = in_features + self.ffn(self.ln2(in_features))
        return in_features


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=None,
        dtype=None,
        attention_backend: str = "reference",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.context_length = context_length
        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

    def allocate_kv_cache(self, *, batch_size, max_seq_len=None):
        """Allocate request-owned storage on the model's device and dtype."""
        limit = self.context_length if max_seq_len is None else max_seq_len
        if type(limit) is not int or not 1 <= limit <= self.context_length:
            raise ValueError("max_seq_len must be within the model context window")
        return KVCache([
            LayerKVCache(
                batch_size=batch_size, num_heads=block.attn.num_heads,
                max_seq_len=limit, head_dim=block.attn.d_heads,
                device=block.attn.k_proj.weight.device, dtype=block.attn.k_proj.weight.dtype,
            )
            for block in self.layers
        ])

    def _validate_kv_cache(self, in_indices, token_positions, cache):
        if not torch.is_inference_mode_enabled():
            raise ValueError("KV cache requires torch.inference_mode()")
        if any(module.training for module in self.modules()):
            raise ValueError("KV cache requires eval mode")
        if token_positions is not None:
            raise ValueError("token_positions are derived from KV cache current_length")
        if not isinstance(cache, KVCache) or len(cache.layers) != self.num_layers:
            raise ValueError("KV cache must have one entry per model layer")
        if in_indices.ndim != 2 or in_indices.shape[1] == 0:
            raise ValueError("cached forward requires nonempty [batch, sequence] input")
        length = cache.current_length
        if length and in_indices.shape[1] != 1:
            raise ValueError("cached decode requires a single new token")
        if cache.max_seq_len > self.context_length:
            raise ValueError("KV cache max_seq_len exceeds model context length")
        for block, layer in zip(self.layers, cache.layers, strict=True):
            if layer.max_seq_len != cache.max_seq_len:
                raise ValueError("KV cache max_seq_len differs between layers")
            if in_indices.device != block.attn.k_proj.weight.device:
                raise ValueError("input device does not match model")
            layer.validate(
                batch_size=in_indices.shape[0], num_heads=block.attn.num_heads,
                head_dim=block.attn.d_heads, device=in_indices.device,
                dtype=block.attn.k_proj.weight.dtype, new_length=in_indices.shape[1],
            )

    def forward(
        self,
        in_indices,
        token_positions: torch.Tensor | None = None,
        *,
        kv_cache: KVCache | None = None,
    ):
        """

        :param in_indices:
        :return:
        """

        if kv_cache is not None:
            self._validate_kv_cache(in_indices, token_positions, kv_cache)
        else:
            if token_positions is not None:
                token_positions = token_positions[..., : self.context_length]
            in_indices = in_indices[..., : self.context_length]

        try:
            in_indices = self.token_embeddings(in_indices)
            for i in range(self.num_layers):
                in_indices = self.layers[i](
                    in_indices,
                    token_positions=token_positions,
                    **({} if kv_cache is None else {"kv_cache": kv_cache.layers[i]}),
                )

            in_indices = self.ln_final(in_indices)
            return self.lm_head(in_indices)
        except Exception:
            # A failed layer can leave earlier layers advanced. Invalidate the
            # request rather than allow a partially updated cache to be reused.
            if kv_cache is not None:
                kv_cache.reset()
            raise
