"""Regenerate traceable Phase 4 tables and charts from retained raw JSON."""

import argparse
from pathlib import Path

from lmforge.benchmarking.inference_results import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    publish(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
