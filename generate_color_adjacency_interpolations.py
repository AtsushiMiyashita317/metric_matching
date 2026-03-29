from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "generated_previews" / "color_adjacency_interpolations"
COLOR_FACTOR_NAMES = ("floor_hue", "wall_hue", "object_hue")


@dataclass(frozen=True)
class InterpolationCase:
    factor_name: str
    factor_index: int
    start_value: float
    end_value: float
    start_index: int
    end_index: int
    dataset_index_start: int
    dataset_index_end: int
    fixed_factors: list[float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate linear interpolation samples between adjacent 3dshapes images where "
            "exactly one color factor differs."
        )
    )
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "3dshapes.h5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Interpolation coefficients. 0=start image, 1=end image.",
    )
    parser.add_argument(
        "--base-configs",
        type=str,
        nargs="*",
        default=[
            "0.0,0.0,0.0,0.75,0,0",
            "0.5,0.0,0.0,0.75,0,0",
            "0.0,0.5,0.0,0.75,1,0",
            "0.0,0.0,0.5,1.25,2,0",
            "0.2,0.4,0.6,1.0357142857142858,3,12.857142857142858",
        ],
        help="Comma-separated factor values [floor_hue, wall_hue, object_hue, scale, shape, orientation].",
    )
    return parser


def parse_base_configs(raw_configs: list[str]) -> list[np.ndarray]:
    parsed: list[np.ndarray] = []
    for raw in raw_configs:
        values = np.array([float(part) for part in raw.split(",")], dtype=np.float64)
        if values.shape != (6,):
            raise ValueError(f"Expected 6 comma-separated values, got {raw!r}")
        parsed.append(values)
    return parsed


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


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


def find_matching_index(labels: np.ndarray, target: np.ndarray) -> int:
    mask = np.ones(labels.shape[0], dtype=bool)
    for factor_idx, value in enumerate(target):
        mask &= np.isclose(labels[:, factor_idx], value)
    matches = np.flatnonzero(mask)
    if matches.size == 0:
        raise ValueError(f"No sample found for factors {target.tolist()}")
    return int(matches[0])


def build_cases(labels: np.ndarray, base_configs: list[np.ndarray]) -> list[InterpolationCase]:
    unique_color_values = [np.unique(labels[:, factor_idx]) for factor_idx in range(3)]
    cases: list[InterpolationCase] = []
    for base_config in base_configs:
        for factor_idx, factor_name in enumerate(COLOR_FACTOR_NAMES):
            values = unique_color_values[factor_idx]
            start_index = int(np.where(np.isclose(values, base_config[factor_idx]))[0][0])
            if start_index + 1 >= len(values):
                continue
            end_index = start_index + 1
            start_factors = base_config.copy()
            end_factors = base_config.copy()
            start_factors[factor_idx] = values[start_index]
            end_factors[factor_idx] = values[end_index]
            cases.append(
                InterpolationCase(
                    factor_name=factor_name,
                    factor_index=factor_idx,
                    start_value=float(values[start_index]),
                    end_value=float(values[end_index]),
                    start_index=start_index,
                    end_index=end_index,
                    dataset_index_start=find_matching_index(labels, start_factors),
                    dataset_index_end=find_matching_index(labels, end_factors),
                    fixed_factors=[float(v) for v in base_config.tolist()],
                )
            )
    return cases


def interpolate_images(start_image: np.ndarray, end_image: np.ndarray, alphas: list[float]) -> list[np.ndarray]:
    start = start_image.astype(np.float32)
    end = end_image.astype(np.float32)
    outputs: list[np.ndarray] = []
    for alpha in alphas:
        blended = (1.0 - alpha) * start + alpha * end
        outputs.append(to_uint8_image(blended))
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    base_configs = parse_base_configs(args.base_configs)

    with h5py.File(args.data_path, "r") as h5_file:
        images = np.array(h5_file["images"], dtype=np.uint8, copy=True)
        labels = np.array(h5_file["labels"], dtype=np.float64, copy=True)

    cases = build_cases(labels, base_configs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, object]] = []
    canvas_rows: list[list[np.ndarray]] = []

    for case_idx, case in enumerate(cases):
        start_image = images[case.dataset_index_start]
        end_image = images[case.dataset_index_end]
        interpolated = interpolate_images(start_image, end_image, args.alphas)
        canvas_rows.append(interpolated)

        image_paths: list[str] = []
        case_dir = args.out_dir / f"case_{case_idx:02d}_{case.factor_name}_{case.start_index}_{case.end_index}"
        for alpha, image in zip(args.alphas, interpolated, strict=True):
            output_path = case_dir / f"alpha_{alpha:.2f}.png"
            save_image(image, output_path)
            image_paths.append(str(output_path.resolve()))

        manifest_cases.append(
            {
                **asdict(case),
                "alphas": args.alphas,
                "image_paths": image_paths,
            }
        )

    comparison_canvas = build_canvas(canvas_rows)
    comparison_path = args.out_dir / "comparison_grid.png"
    save_image(comparison_canvas, comparison_path)

    metadata = {
        "data_path": str(args.data_path.resolve()),
        "definition": {
            "neighbor": "samples that differ in exactly one color factor",
            "adjacent": "neighbor samples whose differing color factor indices differ by 1",
            "interpolation": "pixel-wise linear interpolation in image space",
        },
        "alphas": args.alphas,
        "base_configs": [config.tolist() for config in base_configs],
        "comparison_grid": str(comparison_path.resolve()),
        "cases": manifest_cases,
    }
    metadata_path = args.out_dir / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
