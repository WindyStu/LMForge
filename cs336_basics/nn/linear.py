import math

import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            device=None,
            dtype=None,
    ):
        super(Linear, self).__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2/(in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """

        :param x:
        :return:
        """
        return einsum(self.weight,
                      x,
                      "d_out d_in , ... d_in -> ... d_out",
                      )


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            device=None,
            dtype=None,
    ):
        """
        num_embeddings: int Size of the vocabulary
        19
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super(Embedding, self).__init__()

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight,
                              mean=0,
                              std=1,
                              a=-3,
                              b=3,
                              )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        """
        Lookup the embedding vectors for the given token IDs.
        :param token_ids:
        :return:
        """

        return self.weight[token_ids]