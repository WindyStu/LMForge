import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device=None,
            dtype=None,
    ):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            ))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm 对每个 token 的 hidden dimension 做归一化
        :param x:
        :return:
        """
        input_dtype = x.dtype

        # in case square value overflow
        x_fp32 = x.to(torch.float32)
        rms_value = torch.rsqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        rms_result = x_fp32 * self.weight * rms_value

        # return type of original
        return rms_result.to(input_dtype)
