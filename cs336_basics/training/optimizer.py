import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,):

        defaults = {"lr": lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p.data))
                n = state.get("n", torch.zeros_like(p.data))
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                m = betas[0] * m + (1 - betas[0]) * grad
                n = betas[1] * n + (1 - betas[1]) * grad ** 2
                t += 1

                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                data_t = lr_t * m / (torch.sqrt(n) + eps)
                p.data -= lr * weight_decay * p.data
                p.data -= data_t

                state["t"] = t # Increment iteration number.
                state["m"] = m
                state["n"] = n

        return loss