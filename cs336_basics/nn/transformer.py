import torch
import torch.nn as nn

from cs336_basics.nn.attention import MultiHeadSelfAttention
from cs336_basics.nn.ffn import SwiGLU
from cs336_basics.nn.linear import Embedding, Linear
from cs336_basics.nn.norm import RMSNorm
from cs336_basics.nn.rope import RotaryPositionalEmbedding
from torch import Tensor

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            d_ff,
            max_seq_len,
            theta,
            device=None,
            dtype=None
    ):
        super(TransformerBlock, self).__init__()
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
        )
        self.ffn = SwiGLU(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.ln1  = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.ln2  = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )


    def forward(self, in_features):
        """

        :param in_features:
        :return:
        """
        # 不使用 +=，避免修改调用者 Tensor 和 autograd 问题。
        in_features = in_features + self.attn(self.ln1(in_features))
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
    ):
        super(TransformerLM, self).__init__()
        self.num_layers = num_layers
        self.context_length = context_length
        self.token_embeddings  = Embedding(
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
                ) for _ in range(num_layers)])
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

    def forward(
            self,
            in_indices,
            token_positions: torch.Tensor | None = None,
    ):
        """

        :param in_indices:
        :return:
        """

        if token_positions is not None:
            token_positions = token_positions[
                ..., : self.context_length
            ]


        in_indices = in_indices[..., :self.context_length]

        in_indices = self.token_embeddings(in_indices)
        for i in range(self.num_layers):
            in_indices = self.layers[i](in_indices)

        in_indices = self.ln_final(in_indices)
        in_indices = self.lm_head(in_indices)

        return in_indices
