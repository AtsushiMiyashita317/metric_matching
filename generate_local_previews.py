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

from metric_matching.data import Shapes3DDataModule
from metric_matching.lightning_module import MetricMatchingConfig, MetricMatchingModule


DEFAULT_RUN_DIR = PROJECT_ROOT / "wandb" / "latest-run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the two W&B preview image types locally from the latest run checkpoint."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "3dshapes.h5")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "generated_previews" / "latest_run")
    parser.add_argument("--num-epsilons", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--stats-samples", type=int, default=8192)
    parser.add_argument("--smoothing-sigma", type=float, default=0.6)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def resolve_run_dir(run_dir: Path) -> Path:
    return run_dir.resolve()


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve()
    checkpoint_dir = run_dir / "files" / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    return candidates[-1].resolve()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[MetricMatchingModule, MetricMatchingConfig]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = MetricMatchingConfig(**checkpoint["hyper_parameters"])
    model = MetricMatchingModule(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def build_datamodule(args: argparse.Namespace) -> Shapes3DDataModule:
    datamodule = Shapes3DDataModule(
        data_path=str(args.data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        stats_samples=args.stats_samples,
        smoothing_sigma=args.smoothing_sigma,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    return datamodule


def denormalize(image: torch.Tensor, datamodule: Shapes3DDataModule) -> torch.Tensor:
    stats = datamodule.stats
    if stats is None:
        return image
    mean = stats.mean.to(device=image.device, dtype=image.dtype)
    std = stats.std.to(device=image.device, dtype=image.dtype)
    return image * std + mean


def save_canvas(canvas: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def build_vector_fields_canvas(
    model: MetricMatchingModule,
    datamodule: Shapes3DDataModule,
    epsilon_value: float,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    val_dataset = datamodule.val_dataset
    assert val_dataset is not None
    num_samples = min(model.config.preview_samples, len(val_dataset))
    samples = [val_dataset[idx]["image"] for idx in range(num_samples)]
    normalized_batch = torch.stack(samples, dim=0).to(device)
    epsilon = torch.full((num_samples,), epsilon_value, device=device, dtype=normalized_batch.dtype)

    with torch.no_grad():
        basis_fields = model(normalized_batch, epsilon)
        eigenvectors, eigenvalues = model._top_metric_eigenvectors(basis_fields)

    num_fields = min(model.config.preview_fields, eigenvectors.shape[1])
    rows: list[list[torch.Tensor]] = []
    rows.append([denormalize(image, datamodule).clamp(0.0, 1.0).cpu() for image in normalized_batch])

    displayed_eigenvalues = []
    for field_idx in range(num_fields):
        row_fields = eigenvectors[:, field_idx]
        row_scale = row_fields.pow(2).mean(dim=(1, 2, 3)).sqrt().max().clamp_min(1e-6)
        row_images = [
            model._visualize_vector_field(row_fields[sample_idx], row_scale).cpu()
            for sample_idx in range(num_samples)
        ]
        rows.append(row_images)
        displayed_eigenvalues.append(eigenvalues[:, field_idx].mean().item())

    canvas = model._build_preview_canvas(rows)
    caption = (
        f"top row=input images, lower rows=top metric eigenvectors 0..{num_fields - 1}, "
        f"cols=validation samples 0..{num_samples - 1}, epsilon={epsilon_value:.6g}, "
        f"mean_eigenvalues={[round(v, 4) for v in displayed_eigenvalues]}"
    )
    return canvas, caption


def build_vector_field_integrations_canvas(
    model: MetricMatchingModule,
    datamodule: Shapes3DDataModule,
    epsilon_value: float,
    device: torch.device,
) -> tuple[np.ndarray, str]:
    val_dataset = datamodule.val_dataset
    assert val_dataset is not None
    sample = val_dataset[0]
    normalized_image = sample["image"].unsqueeze(0).to(device)
    epsilon = torch.full((1,), epsilon_value, device=device, dtype=normalized_image.dtype)

    with torch.no_grad():
        eigenvectors, eigenvalues = model._top_metric_eigenvectors_single(normalized_image, epsilon)

    num_fields = min(model.config.preview_fields, eigenvectors.shape[0])
    time_values = torch.linspace(
        -5.0,
        5.0,
        model.config.preview_steps,
        device=device,
        dtype=normalized_image.dtype,
    )

    rows: list[list[torch.Tensor]] = []
    with torch.no_grad():
        for field_idx in range(num_fields):
            vector_field = eigenvectors[field_idx]
            rms = vector_field.pow(2).mean().sqrt().clamp_min(1e-6)
            scale_factor = torch.as_tensor(
                model.config.preview_scale,
                device=device,
                dtype=normalized_image.dtype,
            ) / rms
            row_images = []
            for t in time_values:
                integrated = model._integrate_basis_field_rk4(
                    base_image=normalized_image,
                    epsilon=epsilon,
                    field_idx=field_idx,
                    target_time=float(t.item()),
                    scale_factor=scale_factor,
                    reference_eigenvectors=eigenvectors,
                )[0]
                transformed = denormalize(integrated, datamodule).clamp(0.0, 1.0)
                row_images.append(transformed.cpu())
            rows.append(row_images)

    canvas = model._build_preview_canvas(rows)
    caption = (
        f"rows=tracked metric eigenvectors 0..{num_fields - 1}, "
        f"cols=t in [{time_values[0].item():.2f}, {time_values[-1].item():.2f}], "
        f"epsilon={epsilon_value:.6g}, rk4_substeps={model.config.preview_rk4_substeps}, "
        f"initial_eigenvalues={[round(v.item(), 4) for v in eigenvalues[:num_fields]]}"
    )
    return canvas, caption


def main() -> None:
    args = build_parser().parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    device = torch.device(args.device)

    model, config = load_model(checkpoint_path, device)
    datamodule = build_datamodule(args)

    epsilons = torch.linspace(config.epsilon_min, config.epsilon_max, args.num_epsilons).tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str | float]] = []
    for idx, epsilon_value in enumerate(epsilons):
        vector_fields_canvas, vector_fields_caption = build_vector_fields_canvas(
            model=model,
            datamodule=datamodule,
            epsilon_value=epsilon_value,
            device=device,
        )
        vector_field_integrations_canvas, vector_field_integrations_caption = build_vector_field_integrations_canvas(
            model=model,
            datamodule=datamodule,
            epsilon_value=epsilon_value,
            device=device,
        )

        epsilon_tag = f"{epsilon_value:.6g}".replace(".", "_")
        vector_fields_path = args.out_dir / f"vector_fields_eps{idx:02d}_{epsilon_tag}.png"
        vector_field_integrations_path = (
            args.out_dir / f"vector_field_integrations_eps{idx:02d}_{epsilon_tag}.png"
        )
        save_canvas(vector_fields_canvas, vector_fields_path)
        save_canvas(vector_field_integrations_canvas, vector_field_integrations_path)
        manifest.extend(
            [
                {
                    "type": "vector_fields",
                    "epsilon": epsilon_value,
                    "path": str(vector_fields_path.resolve()),
                    "caption": vector_fields_caption,
                },
                {
                    "type": "vector_field_integrations",
                    "epsilon": epsilon_value,
                    "path": str(vector_field_integrations_path.resolve()),
                    "caption": vector_field_integrations_caption,
                },
            ]
        )

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "epsilon_min": config.epsilon_min,
        "epsilon_max": config.epsilon_max,
        "num_epsilons": args.num_epsilons,
        "smoothing_sigma": args.smoothing_sigma,
        "outputs": manifest,
    }
    metadata_path = args.out_dir / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
