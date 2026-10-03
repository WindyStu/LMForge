"""Parity contracts for configurable attention backends."""

from __future__ import annotations

import copy

import torch

from lmforge.nn.attention import MultiHeadSelfAttention
from lmforge.nn.transformer import TransformerLM
from lmforge.training.evaluation import language_model_loss


def _attention(backend: str) -> MultiHeadSelfAttention:
    return MultiHeadSelfAttention(
        d_model=16,
        num_heads=4,
        use_rope=True,
        max_seq=8,
        theta=10_000.0,
        attention_backend=backend,
    )


def test_sdpa_attention_matches_reference_forward_and_gradients() -> None:
    torch.manual_seed(7)
    reference = _attention("reference")
    sdpa = _attention("sdpa")
    sdpa.load_state_dict(reference.state_dict())
    reference_input = torch.randn(2, 8, 16, requires_grad=True)
    sdpa_input = reference_input.detach().clone().requires_grad_(True)
    positions = torch.arange(8).expand(2, -1)

    reference_output = reference(reference_input, token_positions=positions)
    sdpa_output = sdpa(sdpa_input, token_positions=positions)
    reference_output.square().sum().backward()
    sdpa_output.square().sum().backward()

    torch.testing.assert_close(sdpa_output, reference_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(sdpa_input.grad, reference_input.grad, rtol=5e-5, atol=5e-6)
    for (reference_name, reference_parameter), (sdpa_name, sdpa_parameter) in zip(
        reference.named_parameters(), sdpa.named_parameters(), strict=True
    ):
        assert sdpa_name == reference_name
        torch.testing.assert_close(
            sdpa_parameter.grad,
            reference_parameter.grad,
            rtol=5e-5,
            atol=5e-6,
        )


def test_sdpa_transformer_matches_reference_logits_loss_and_gradients() -> None:
    torch.manual_seed(11)
    arguments = {
        "vocab_size": 32,
        "context_length": 8,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 32,
        "rope_theta": 10_000.0,
    }
    reference = TransformerLM(**arguments, attention_backend="reference")
    sdpa = TransformerLM(**arguments, attention_backend="sdpa")
    sdpa.load_state_dict(reference.state_dict())
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))

    reference_logits = reference(inputs)
    sdpa_logits = sdpa(inputs)
    reference_loss = language_model_loss(reference, inputs, targets)
    sdpa_loss = language_model_loss(sdpa, inputs, targets)
    reference_loss.backward()
    sdpa_loss.backward()

    torch.testing.assert_close(sdpa_logits, reference_logits, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(sdpa_loss, reference_loss, rtol=2e-6, atol=2e-6)
    for reference_parameter, sdpa_parameter in zip(reference.parameters(), sdpa.parameters(), strict=True):
        torch.testing.assert_close(
            sdpa_parameter.grad,
            reference_parameter.grad,
            rtol=8e-5,
            atol=8e-6,
        )


def test_sdpa_attention_is_causal() -> None:
    torch.manual_seed(13)
    attention = _attention("sdpa")
    inputs = torch.randn(1, 8, 16)
    changed = copy.deepcopy(inputs)
    changed[:, 5:] = torch.randn_like(changed[:, 5:])

    original_output = attention(inputs)
    changed_output = attention(changed)

    torch.testing.assert_close(changed_output[:, :5], original_output[:, :5])


def test_reference_and_sdpa_state_dicts_are_interchangeable() -> None:
    reference = _attention("reference")
    sdpa = _attention("sdpa")

    sdpa.load_state_dict(reference.state_dict(), strict=True)
    reference.load_state_dict(sdpa.state_dict(), strict=True)

    assert reference.state_dict().keys() == sdpa.state_dict().keys()
