from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from PIL import Image

from metric_matching.data import Shapes3DDataset, apply_gaussian_smoothing, restore_image_range


DEFAULT_OUT_DIR = PROJECT_ROOT / "generated_previews" / "smoothing_samples"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save local sample images showing the effect of different smoothing sigmas."
    )
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "3dshapes.h5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[12000, 12010, 12100, 13000])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9, 1.2])
    return parser


def to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = restore_image_range(image.detach())
    return image.clamp(0.0, 1.0).permute(1, 2, 0).mul(255).byte().cpu().numpy()


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_canvas(rows: list[list[np.ndarray]], gap: int = 4) -> np.ndarray:
    row_count = len(rows)
    col_count = len(rows[0])
    height, width, channels = rows[0][0].shape
    canvas = np.ones(
        (
            row_count * height + (row_count - 1) * gap,
            col_count * width + (col_count - 1) * gap,
            channels,
        ),
        dtype=np.uint8,
    ) * 255
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            y0 = row_idx * (height + gap)
            x0 = col_idx * (width + gap)
            canvas[y0 : y0 + height, x0 : x0 + width] = image
    return canvas


def main() -> None:
    args = build_parser().parse_args()
    dataset = Shapes3DDataset(args.data_path, normalize=False, smoothing_sigma=0.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, int | float | str]] = []
    canvas_rows: list[list[np.ndarray]] = []
    sigma_labels = ["original" if sigma == 0.0 else f"sigma_{sigma:.2f}" for sigma in args.sigmas]

    for sample_index in args.sample_indices:
        sample = dataset[sample_index]
        base_image = sample["image"]
        row: list[np.ndarray] = []
        for sigma, sigma_label in zip(args.sigmas, sigma_labels, strict=True):
            smoothed = apply_gaussian_smoothing(base_image, sigma=sigma)
            image_np = to_uint8_image(smoothed)
            output_path = args.out_dir / f"sample_{sample_index:05d}_{sigma_label}.png"
            save_image(image_np, output_path)
            row.append(image_np)
            manifest.append(
                {
                    "sample_index": sample_index,
                    "sigma": sigma,
                    "label": sigma_label,
                    "path": str(output_path.resolve()),
                }
            )
        canvas_rows.append(row)

    comparison_canvas = build_canvas(canvas_rows)
    comparison_path = args.out_dir / "comparison_grid.png"
    save_image(comparison_canvas, comparison_path)

    metadata = {
        "data_path": str(args.data_path.resolve()),
        "sample_indices": args.sample_indices,
        "sigmas": args.sigmas,
        "column_labels": sigma_labels,
        "comparison_grid": str(comparison_path.resolve()),
        "outputs": manifest,
    }
    metadata_path = args.out_dir / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
