import math

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Bool, Float
from torch import Tensor, nn

from .kv_cache import LayerKVCache
from .linear import Linear
from .rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # compute Q * K^T
    scale = 1.0 / math.sqrt(Q.shape[-1])

    scores = einsum(
        Q,
        K,
        "... queries d_k, ... keys d_k -> ... queries keys",
    )

    # 标量可以直接和tensor做运算 底层隐式升级为0维tensor
    scores = scores * scale

    # True 表示允许参与attention
    # -1e9 对 fp16 可能溢出；应使用 masked_fill(~mask, -torch.inf)
    if mask is not None:
        scores = scores.masked_fill(~mask.to(device=scores.device), float("-inf"))

    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    return einsum(
        probabilities,
        V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        use_rope: bool = False,
        max_seq: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
        attention_backend: str = "reference",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads
        if attention_backend not in {"reference", "naive", "sdpa"}:
            raise ValueError("attention_backend must be reference, naive or sdpa")
        self.attention_backend = attention_backend
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_heads, max_seq, device=device)
        else:
            self.rope = None
        self.q_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            d_model,
            d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Tensor | None = None,
        *,
        kv_cache: LayerKVCache | None = None,
    ):
        """

        :param in_features:
        :return:
        """
        position_offset = 0
        if kv_cache is not None:
            if not torch.is_inference_mode_enabled():
                raise ValueError("KV cache requires torch.inference_mode()")
            if self.training:
                raise ValueError("KV cache requires eval mode")
            if token_positions is not None:
                raise ValueError("token_positions are derived from KV cache current_length")
            if in_features.ndim != 3:
                raise ValueError("cached attention requires [batch, sequence, features]")
            position_offset = kv_cache.current_length
            if position_offset and in_features.shape[-2] != 1:
                raise ValueError("cached decode requires a single new token")
            kv_cache.validate(
                batch_size=in_features.shape[0], num_heads=self.num_heads, head_dim=self.d_heads,
                device=in_features.device, dtype=self.k_proj.weight.dtype,
                new_length=in_features.shape[-2],
            )
        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)

        q_head = rearrange(q, "... seq (nums_head  d)-> ... nums_head seq d", nums_head=self.num_heads)
        k_head = rearrange(k, "... seq (nums_head  d) -> ... nums_head seq d", nums_head=self.num_heads)
        v_head = rearrange(v, "... seq (nums_head  d) -> ... nums_head seq d", nums_head=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(position_offset, position_offset + in_features.shape[-2], device=in_features.device).expand(
                    in_features.shape[:-1]
                )
            # [batch, seq] -> [batch, 1, seq]
            # 增加 head 维，才能和 [batch, heads, seq, head_dim] 广播。
            rope_positions = token_positions.to(in_features.device).unsqueeze(-2)

            q_head = self.rope(q_head, rope_positions)
            k_head = self.rope(k_head, rope_positions)

        if kv_cache is not None:
            k_head, v_head = kv_cache.append(k_head, v_head)
        # With one newest query, every stored key is visible. SDPA's ordinary
        # non-square causal mask would incorrectly expose only the first key.
        is_causal = position_offset == 0
        if self.attention_backend == "sdpa":
            attended = F.scaled_dot_product_attention(
                q_head,
                k_head,
                v_head,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        else:
            seq_len = in_features.shape[-2]
            mask = None
            if is_causal:
                mask = torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=in_features.device,
                ).tril()
            attended = scaled_dot_product_attention(q_head, k_head, v_head, mask)
        multi_head = rearrange(
            attended,
            "... nums_head seq d -> ... seq (nums_head d)",
        )

        return self.output_proj(multi_head)
