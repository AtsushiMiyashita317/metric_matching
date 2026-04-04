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

from metric_matching.adversarial_module import (
    AdversarialMetricConfig,
    AdversarialMetricModule,
)
from metric_matching.data import Shapes3DDataModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an adversarial denoiser and structured noise generator on 3dshapes."
    )
    parser.add_argument("--data-path", type=str, default="3dshapes.h5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--denoiser-learning-rate", type=float, default=2e-3)
    parser.add_argument("--generator-learning-rate", type=float, default=2e-5)
    parser.add_argument("--denoiser-lr-alpha", type=float, default=0.666)
    parser.add_argument("--generator-lr-alpha", type=float, default=1.0)
    parser.add_argument("--denoiser-warmup-steps", type=int, default=0)
    parser.add_argument("--generator-warmup-steps", type=int, default=5000)
    parser.add_argument("--denoiser-lr-scale-steps", type=int, default=5000)
    parser.add_argument("--generator-lr-scale-steps", type=int, default=5000)
    parser.add_argument("--denoiser-weight-decay", type=float, default=0.0)
    parser.add_argument("--generator-weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-downsample-factor", type=int, default=4)
    parser.add_argument("--disable-output-bias", action="store_true")
    parser.add_argument("--output-bias-variance", type=float, default=1e-3)
    parser.add_argument("--epsilon-min", type=float, default=1e-4)
    parser.add_argument("--epsilon-max", type=float, default=1e-2)
    parser.add_argument("--generator-loss-weight", type=float, default=1.0)
    parser.add_argument("--covariance-regularization", type=float, default=1.0)
    parser.add_argument("--scale-input", action="store_true")
    parser.add_argument(
        "--eps-input-mode",
        type=str,
        default="log_clamp",
        choices=["log_clamp", "log_one_plus", "identity"],
    )
    parser.add_argument("--preview-samples", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--disable-normalize", action="store_true")
    parser.add_argument("--stats-samples", type=int, default=8192)
    parser.add_argument("--smoothing-sigma", type=float, default=0.6)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--enable-color-interpolation", action="store_true")
    parser.add_argument("--project", type=str, default="metric-matching")
    parser.add_argument("--run-name", type=str, default="3dshapes-adversarial-denoiser")
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
        normalize=not args.disable_normalize,
        stats_samples=args.stats_samples,
        smoothing_sigma=args.smoothing_sigma,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        enable_color_interpolation=args.enable_color_interpolation,
        seed=args.seed,
    )
    data_module.prepare_data()
    data_module.setup("fit")
    if data_module.image_shape is None:
        raise RuntimeError("DataModule did not expose image shape.")

    channels, height, width = data_module.image_shape
    if height != width:
        raise ValueError(f"Only square images are supported, received {(channels, height, width)}")

    config = AdversarialMetricConfig(
        image_channels=channels,
        image_size=height,
        rank=args.rank,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        attention_downsample_factor=args.attention_downsample_factor,
        use_output_bias=not args.disable_output_bias,
        output_bias_variance=args.output_bias_variance,
        denoiser_learning_rate=args.denoiser_learning_rate,
        generator_learning_rate=args.generator_learning_rate,
        denoiser_lr_alpha=args.denoiser_lr_alpha,
        generator_lr_alpha=args.generator_lr_alpha,
        denoiser_warmup_steps=args.denoiser_warmup_steps,
        generator_warmup_steps=args.generator_warmup_steps,
        denoiser_lr_scale_steps=args.denoiser_lr_scale_steps,
        generator_lr_scale_steps=args.generator_lr_scale_steps,
        denoiser_weight_decay=args.denoiser_weight_decay,
        generator_weight_decay=args.generator_weight_decay,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        generator_loss_weight=args.generator_loss_weight,
        covariance_regularization=args.covariance_regularization,
        scale_input=args.scale_input,
        epsilon_input_mode=args.eps_input_mode,
        preview_samples=args.preview_samples,
    )
    model = AdversarialMetricModule(config)

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
                "normalize": not args.disable_normalize,
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
            filename="adversarial-denoiser-{epoch:02d}",
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
        # gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=False,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
