import torch
import torch.nn as nn

from .linear import Linear
from einops import rearrange, einsum

def silu(x: torch.Tensor) -> torch.Tensor:
    """

    :param x:
    :return:
    """
    # sigmoid 比显式 exp(-x) 更稳定。
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model,
            d_ff,
            device=None,
            dtype=None,
    ):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            d_ff,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.w3 = Linear(
            d_model,
            d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_features):
        """

        :param in_features:
        :return:
        """
        gate = silu(self.w1(in_features))
        value = self.w3(in_features)

        return self.w2(gate * value)
