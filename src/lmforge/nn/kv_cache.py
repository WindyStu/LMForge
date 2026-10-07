"""Preallocated, request-owned KV storage; never registered in model state."""

from __future__ import annotations

import torch


class LayerKVCache:
    """One layer's RoPE-rotated keys and unrotated values, in [B, H, T, D]."""

    def __init__(self, *, batch_size, num_heads, max_seq_len, head_dim, device, dtype):
        for name, value in (
            ("batch_size", batch_size),
            ("num_heads", num_heads),
            ("max_seq_len", max_seq_len),
            ("head_dim", head_dim),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype
        self.current_length = 0
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.key = torch.empty(shape, device=self.device, dtype=dtype)
        self.value = torch.empty_like(self.key)
        self.device = self.key.device

    def reset(self):
        """Reuse storage. Unwritten/stale slots are excluded from attention."""
        self.current_length = 0

    def validate(self, *, batch_size, num_heads, head_dim, device, dtype, new_length):
        if self.batch_size != batch_size:
            raise ValueError("KV cache batch size does not match input")
        if self.device != device:
            raise ValueError("KV cache device does not match model/input")
        if self.dtype != dtype:
            raise ValueError("KV cache dtype does not match projections")
        shape = (batch_size, num_heads, self.max_seq_len, head_dim)
        for storage in (self.key, self.value):
            if storage.shape != shape or storage.device != device or storage.dtype != dtype:
                raise ValueError("KV cache storage shape/device/dtype is incompatible")
        if type(self.current_length) is not int or not 0 <= self.current_length <= self.max_seq_len:
            raise ValueError("invalid KV cache current_length")
        if self.current_length + new_length > self.max_seq_len:
            raise ValueError("KV cache exceeds max_seq_len")

    def append(self, key, value):
        if not torch.is_inference_mode_enabled():
            raise ValueError("KV cache requires torch.inference_mode()")
        if key.ndim != 4 or key.shape != value.shape or key.device != value.device or key.dtype != value.dtype:
            raise ValueError("key/value shape, device and dtype must match")
        self.validate(
            batch_size=key.shape[0],
            num_heads=key.shape[1],
            head_dim=key.shape[3],
            device=key.device,
            dtype=key.dtype,
            new_length=key.shape[2],
        )
        start, end = self.current_length, self.current_length + key.shape[2]
        self.key[:, :, start:end].copy_(key)
        self.value[:, :, start:end].copy_(value)
        self.current_length = end
        return self.key[:, :, :end], self.value[:, :, :end]


class KVCache:
    """Independent per-layer storage for a single equal-length batch."""

    def __init__(self, layers):
        if not layers:
            raise ValueError("KV cache requires at least one layer")
        self.layers = tuple(layers)
        first = self.layers[0]
        self.batch_size = first.batch_size
        self.max_seq_len = first.max_seq_len
        self.device = first.device
        self.dtype = first.dtype

    @property
    def current_length(self):
        length = self.layers[0].current_length
        if any(layer.current_length != length for layer in self.layers):
            raise ValueError("KV cache layer lengths are inconsistent; reset before reuse")
        return length

    def reset(self):
        """Reset every layer without reallocating K/V."""
        for layer in self.layers:
            layer.reset()
