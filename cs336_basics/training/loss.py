from collections.abc import Iterable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def softmax(
        in_features: Float[Tensor, " ..."],
        dim: int
) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    in_features_e = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True).values)
    divide = torch.sum(in_features_e, dim=dim, keepdim=True)
    return in_features_e / divide

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    input_max = torch.max(inputs, dim=-1, keepdim=True).values
    bs = inputs.shape[0]
    x_target = inputs[range(bs), targets].unsqueeze(-1)
    x_sub = inputs - input_max
    ce_loss = input_max - x_target + torch.log(torch.sum(torch.exp(x_sub), dim=-1, keepdim=True))

    return torch.mean(ce_loss)

@torch.no_grad()
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    if max_l2_norm < 0 or eps <= 0:
        raise ValueError("max_l2_norm must be non-negative and eps positive")
    # Materialize gradients once: model.parameters() is a one-shot iterator.
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads]))
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    scale = (max_l2_norm / (norm + eps)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale.to(grad.dtype))
