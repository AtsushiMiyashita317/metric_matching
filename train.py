from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from metric_matching.data import Shapes3DDataModule
from metric_matching.lightning_module import MetricMatchingConfig, MetricMatchingModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Riemannian Metric Matching on 3dshapes.")
    parser.add_argument("--data-path", type=str, default="3dshapes.h5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-downsample-factor", type=int, default=4)
    parser.add_argument("--disable-output-bias", action="store_true")
    parser.add_argument("--output-bias-variance", type=float, default=1e-3)
    parser.add_argument("--epsilon-min", type=float, default=1e-4)
    parser.add_argument("--epsilon-max", type=float, default=5e-2)
    parser.add_argument("--copies-per-sample", type=int, default=1)
    parser.add_argument("--tikhonov-lambda", type=float, default=1e-4)
    parser.add_argument("--preview-fields", type=int, default=8)
    parser.add_argument("--preview-samples", type=int, default=4)
    parser.add_argument("--preview-steps", type=int, default=7)
    parser.add_argument("--preview-scale", type=float, default=0.25)
    parser.add_argument("--preview-rk4-substeps", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--stats-samples", type=int, default=8192)
    parser.add_argument("--smoothing-sigma", type=float, default=0.6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--project", type=str, default="metric-matching")
    parser.add_argument("--run-name", type=str, default="3dshapes-rmm")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--log-model", action="store_true")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    L.seed_everything(args.seed, workers=True)

    data_module = Shapes3DDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        stats_samples=args.stats_samples,
        smoothing_sigma=args.smoothing_sigma,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
    )
    data_module.prepare_data()
    data_module.setup("fit")
    if data_module.image_shape is None:
        raise RuntimeError("DataModule did not expose image shape.")

    channels, height, width = data_module.image_shape
    if height != width:
        raise ValueError(f"Only square images are supported, received {(channels, height, width)}")

    config = MetricMatchingConfig(
        image_channels=channels,
        image_size=height,
        rank=args.rank,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        attention_downsample_factor=args.attention_downsample_factor,
        use_output_bias=not args.disable_output_bias,
        output_bias_variance=args.output_bias_variance,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        copies_per_sample=args.copies_per_sample,
        tikhonov_lambda=args.tikhonov_lambda,
        preview_fields=args.preview_fields,
        preview_samples=args.preview_samples,
        preview_steps=args.preview_steps,
        preview_scale=args.preview_scale,
        preview_rk4_substeps=args.preview_rk4_substeps,
    )
    model = MetricMatchingModule(config)

    logger = None
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            project=args.project,
            name=args.run_name,
            log_model=args.log_model,
            mode=args.wandb_mode,
        )
        logger.experiment.config.update(vars(args))
        logger.experiment.config.update(
            {
                "image_channels": channels,
                "image_height": height,
                "image_width": width,
                "torch_version": torch.__version__,
                "dataset_path": str(Path(args.data_path).resolve()),
                "smoothing_sigma": args.smoothing_sigma,
            }
        )

    checkpoint_dir = None
    if logger is not None:
        checkpoint_dir = Path(logger.experiment.dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir) if checkpoint_dir is not None else None,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="metric-matching-{epoch:02d}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=False,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
