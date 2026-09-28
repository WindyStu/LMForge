"""Render train and validation loss from LMForge metrics JSONL as SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_points(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    validation: list[tuple[int, float]] = []
    previous_step = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        step = record.get("step")
        train_loss = record.get("train_loss")
        if not isinstance(step, int) or step <= previous_step:
            raise ValueError(f"line {line_number}: steps must be strictly increasing")
        if not isinstance(train_loss, (int, float)) or not math.isfinite(train_loss):
            raise ValueError(f"line {line_number}: train_loss must be finite")
        train.append((step, float(train_loss)))
        if "val_loss" in record:
            val_loss = record["val_loss"]
            if not isinstance(val_loss, (int, float)) or not math.isfinite(val_loss):
                raise ValueError(f"line {line_number}: val_loss must be finite")
            validation.append((step, float(val_loss)))
        previous_step = step
    if not train:
        raise ValueError("metrics file contains no training records")
    if not validation:
        raise ValueError("metrics file contains no validation records")
    return train, validation


def render_svg(
    train: list[tuple[int, float]],
    validation: list[tuple[int, float]],
) -> str:
    width, height = 960, 540
    left, right, top, bottom = 80, 30, 45, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = train + validation
    x_min, x_max = train[0][0], train[-1][0]
    y_values = [loss for _, loss in all_points]
    y_min, y_max = min(y_values), max(y_values)
    y_padding = max((y_max - y_min) * 0.05, 1e-9)
    y_min -= y_padding
    y_max += y_padding

    def project(point: tuple[int, float]) -> tuple[float, float]:
        step, loss = point
        x_span = max(x_max - x_min, 1)
        x = left + (step - x_min) / x_span * plot_width
        y = top + (y_max - loss) / (y_max - y_min) * plot_height
        return x, y

    def polyline(points: list[tuple[int, float]], color: str) -> str:
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, points))
        return (
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    grid = []
    labels = []
    for index in range(6):
        fraction = index / 5
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#d9dee7" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="13" fill="#4b5563">{value:.3f}</text>'
        )
    for index in range(6):
        fraction = index / 5
        x = left + fraction * plot_width
        step = round(x_min + fraction * (x_max - x_min))
        labels.append(
            f'<text x="{x:.2f}" y="{height - bottom + 25}" text-anchor="middle" '
            f'font-size="13" fill="#4b5563">{step}</text>'
        )

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            (
                '<text x="480" y="28" text-anchor="middle" font-size="20" '
                'font-family="sans-serif" fill="#111827">TinyStories Training Loss</text>'
            ),
            *grid,
            (f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827"/>'),
            (
                f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
                f'y2="{height - bottom}" stroke="#111827"/>'
            ),
            *labels,
            polyline(train, "#2563eb"),
            polyline(validation, "#dc2626"),
            '<line x1="700" y1="20" x2="730" y2="20" stroke="#2563eb" stroke-width="2"/>',
            '<text x="738" y="25" font-size="13" font-family="sans-serif">train</text>',
            '<line x1="805" y1="20" x2="835" y2="20" stroke="#dc2626" stroke-width="2"/>',
            '<text x="843" y="25" font-size="13" font-family="sans-serif">validation</text>',
            (
                f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-size="14" '
                'font-family="sans-serif">optimizer step</text>'
            ),
            (
                f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="14" '
                'font-family="sans-serif" transform="rotate(-90 20 270)">'
                "cross-entropy loss</text>"
            ),
            "</svg>",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".svg":
        parser.error("--output must use the .svg extension")
    train, validation = load_points(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(train, validation), encoding="utf-8")


if __name__ == "__main__":
    main()
