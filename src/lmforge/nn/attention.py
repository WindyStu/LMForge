import math

import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
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
            device = None,
            dtype = None,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads
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
    ):
        """

        :param in_features:
        :return:
        """
        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)

        q_head = rearrange(q, "... seq (nums_head  d)-> ... nums_head seq d", nums_head = self.num_heads)
        k_head = rearrange(k, "... seq (nums_head  d) -> ... nums_head seq d", nums_head = self.num_heads)
        v_head = rearrange(v, "... seq (nums_head  d) -> ... nums_head seq d", nums_head = self.num_heads)

        if self.rope is not None:
            if  token_positions is None:
                token_positions = torch.arange(
                    in_features.shape[-2], device=in_features.device
                ).expand(in_features.shape[:-1])
            # [batch, seq] -> [batch, 1, seq]
            # 增加 head 维，才能和 [batch, heads, seq, head_dim] 广播。
            rope_positions = token_positions.to(in_features.device).unsqueeze(-2)

            q_head = self.rope(q_head, rope_positions)
            k_head = self.rope(k_head, rope_positions)

        seq_len = in_features.shape[-2]
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device).tril()
        multi_head = rearrange(scaled_dot_product_attention(q_head, k_head, v_head, mask), "... nums_head seq d -> ... seq (nums_head d)")

        return self.output_proj(multi_head)
