from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FACTOR_NAMES = ("floor_hue", "wall_hue", "object_hue")
NON_COLOR_FACTORS = {
    3: 0.75,
    4: 0.0,
    5: 0.0,
}


@dataclass(frozen=True)
class ColorPoint:
    factor_name: str
    factor_value: float
    rgb: tuple[int, int, int]
    pixel_row: int
    pixel_col: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 3dshapes color factors inside RGB space.",
    )
    parser.add_argument("--data-path", type=Path, default=Path("3dshapes.h5"))
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("outputs/3dshapes_color_rgb_space.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/3dshapes_color_rgb_space.csv"),
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(data_path, "r") as h5_file:
        images = np.array(h5_file["images"], copy=True)
        labels = np.array(h5_file["labels"], copy=True)
    return images, labels


def build_mask(labels: np.ndarray, target_factor: int, target_value: float) -> np.ndarray:
    mask = np.isclose(labels[:, target_factor], target_value)
    for factor_index in range(3):
        if factor_index == target_factor:
            continue
        mask &= np.isclose(labels[:, factor_index], 0.0)
    for factor_index, factor_value in NON_COLOR_FACTORS.items():
        mask &= np.isclose(labels[:, factor_index], factor_value)
    return mask


def find_representative_pixel(
    images: np.ndarray,
    labels: np.ndarray,
    target_factor: int,
) -> tuple[int, int]:
    factor_values = np.unique(labels[:, target_factor])
    slices = []
    for factor_value in factor_values:
        matching_indices = np.flatnonzero(build_mask(labels, target_factor, factor_value))
        if matching_indices.size == 0:
            raise ValueError(
                f"Could not find a reference image for factor {target_factor} value {factor_value}."
            )
        slices.append(images[matching_indices[0]].astype(np.float32))

    stacked = np.stack(slices, axis=0)
    pixel_variance = stacked.var(axis=0).sum(axis=-1)
    best_row, best_col = np.unravel_index(np.argmax(pixel_variance), pixel_variance.shape)
    return int(best_row), int(best_col)


def extract_color_points(images: np.ndarray, labels: np.ndarray) -> list[ColorPoint]:
    points: list[ColorPoint] = []
    for factor_index, factor_name in enumerate(FACTOR_NAMES):
        row, col = find_representative_pixel(images, labels, factor_index)
        factor_values = np.unique(labels[:, factor_index])
        for factor_value in factor_values:
            matching_indices = np.flatnonzero(build_mask(labels, factor_index, factor_value))
            image_index = int(matching_indices[0])
            rgb = tuple(int(v) for v in images[image_index, row, col])
            points.append(
                ColorPoint(
                    factor_name=factor_name,
                    factor_value=float(factor_value),
                    rgb=rgb,
                    pixel_row=row,
                    pixel_col=col,
                )
            )
    return points


def rotate(points: np.ndarray, yaw_deg: float = -42.0, pitch_deg: float = 28.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    rot_y = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return points @ rot_y.T @ rot_x.T


def project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    centered = points - 127.5
    rotated = rotate(centered)
    scale = 1.65
    projected = rotated[:, :2] * scale
    projected[:, 0] += width * 0.38
    projected[:, 1] = height * 0.72 - projected[:, 1]
    return projected


def draw_axes(draw: ImageDraw.ImageDraw, width: int, height: int, font: ImageFont.ImageFont) -> None:
    cube_points = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ],
        dtype=np.float32,
    )
    projected = project(cube_points, width, height)
    edges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    )
    for start, end in edges:
        draw.line(
            [tuple(projected[start]), tuple(projected[end])],
            fill=(180, 180, 180),
            width=2,
        )

    origin = tuple(projected[0])
    axis_specs = (
        (1, "R", (225, 60, 60)),
        (2, "G", (40, 160, 70)),
        (3, "B", (60, 110, 220)),
    )
    for point_index, label, color in axis_specs:
        axis_end = tuple(projected[point_index])
        draw.line([origin, axis_end], fill=color, width=4)
        draw.text((axis_end[0] + 8, axis_end[1] - 8), f"{label}=255", fill=color, font=font)
    draw.text((origin[0] - 10, origin[1] + 8), "0", fill=(90, 90, 90), font=font)


def draw_points(
    image: Image.Image,
    points: list[ColorPoint],
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(image)
    rgb_values = np.array([point.rgb for point in points], dtype=np.float32)
    projected = project(rgb_values, width, height)
    depth = rotate(rgb_values - 127.5)[:, 2]
    order = np.argsort(depth)
    for idx in order:
        x, y = projected[idx]
        rgb = tuple(int(v) for v in rgb_values[idx])
        outline = (255, 255, 255) if sum(rgb) < 390 else (30, 30, 30)
        radius = 8
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=rgb,
            outline=outline,
            width=2,
        )

    legend_x = int(width * 0.64)
    legend_y = 90
    for factor_name in FACTOR_NAMES:
        draw.text((legend_x, legend_y), factor_name, fill=(35, 35, 35), font=font)
        legend_y += 28
        factor_points = [point for point in points if point.factor_name == factor_name]
        for point in factor_points:
            swatch_top = legend_y + 3
            draw.rectangle(
                [(legend_x, swatch_top), (legend_x + 18, swatch_top + 18)],
                fill=point.rgb,
                outline=(50, 50, 50),
            )
            text = f"{point.factor_value:.1f} -> {point.rgb}"
            draw.text((legend_x + 28, legend_y), text, fill=(60, 60, 60), font=font)
            legend_y += 24
        legend_y += 18


def render(points: list[ColorPoint], output_path: Path) -> None:
    width, height = 1400, 980
    image = Image.new("RGB", (width, height), (250, 248, 242))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((60, 40), "3dshapes color factors in RGB space", fill=(20, 20, 20), font=font)
    draw.text(
        (60, 62),
        "Representative RGB values extracted from rendered pixels for floor_hue / wall_hue / object_hue",
        fill=(85, 85, 85),
        font=font,
    )

    draw_axes(draw, width, height, font)
    draw_points(image, points, width, height, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_csv(points: list[ColorPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["factor_name", "factor_value", "red", "green", "blue", "pixel_row", "pixel_col"])
        for point in points:
            writer.writerow(
                [
                    point.factor_name,
                    point.factor_value,
                    point.rgb[0],
                    point.rgb[1],
                    point.rgb[2],
                    point.pixel_row,
                    point.pixel_col,
                ]
            )


def main() -> None:
    args = parse_args()
    images, labels = load_dataset(args.data_path)
    points = extract_color_points(images, labels)
    render(points, args.output_image)
    write_csv(points, args.output_csv)
    for factor_name in FACTOR_NAMES:
        factor_points = [point for point in points if point.factor_name == factor_name]
        representative_pixel = (factor_points[0].pixel_row, factor_points[0].pixel_col)
        print(f"{factor_name}: representative pixel {representative_pixel}")
        for point in factor_points:
            print(f"  {point.factor_value:.1f}: {point.rgb}")
    print(f"saved image to {args.output_image}")
    print(f"saved csv to {args.output_csv}")


if __name__ == "__main__":
    main()
