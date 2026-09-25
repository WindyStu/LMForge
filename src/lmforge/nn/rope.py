import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        assert d_k % 2 == 0

        i = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

        k = torch.arange(
            1,
            d_k//2 + 1,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        angle = i / (theta ** ((2 * k - 2) / d_k))

        self.register_buffer(
            "cos_table",
            torch.cos(angle),
            persistent=False,
        )
        self.register_buffer(
            "sin_table",
            torch.sin(angle),
            persistent=False,
        )

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)-> torch.Tensor:
        """

        :param x:
        :param token_positions:
        :return:
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out