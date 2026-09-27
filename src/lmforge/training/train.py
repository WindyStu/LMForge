"""Compatibility exports and legacy module entry point for training."""

from ..config import ModelConfig, TrainConfig
from .engine import train

__all__ = ["ModelConfig", "TrainConfig", "train"]


def main() -> int:
    from ..cli import legacy_main

    return legacy_main("train")


if __name__ == "__main__":
    raise SystemExit(main())
