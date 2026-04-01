from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch

from metric_matching.data import restore_image_range
from metric_matching.models import ScoreNetwork


def prediction_to_noise(
    prediction: torch.Tensor,
    noisy_images: torch.Tensor,
    epsilon: torch.Tensor,
    score_target: Literal["noise", "mean"],
) -> torch.Tensor:
    if score_target == "noise":
        return prediction
    return (noisy_images - prediction) / epsilon.sqrt()[:, None, None, None]


def prediction_to_denoised(
    prediction: torch.Tensor,
    noisy_images: torch.Tensor,
    epsilon: torch.Tensor,
    score_target: Literal["noise", "mean"],
) -> torch.Tensor:
    if score_target == "noise":
        return noisy_images - prediction * epsilon.sqrt()[:, None, None, None]
    return prediction


def _match_prefixed_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    for prefix in prefixes:
        if prefix:
            candidate = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        else:
            candidate = dict(state_dict)
        if candidate:
            return candidate
    raise KeyError(
        "Could not find score predictor weights in the checkpoint. "
        f"Tried prefixes: {prefixes}."
    )


def load_score_network_checkpoint(
    network: ScoreNetwork,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(resolved_path, map_location="cpu")
    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state_dict, dict):
        raise TypeError(f"Checkpoint at {resolved_path} does not contain a valid state_dict.")

    candidate_state_dict = _match_prefixed_state_dict(
        raw_state_dict,
        prefixes=("network.", "score_network.", "model.", ""),
    )
    missing_keys, unexpected_keys = network.load_state_dict(candidate_state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Score predictor checkpoint is incompatible. "
            f"Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}"
        )
    return {
        "checkpoint_path": str(resolved_path),
        "checkpoint_keys": tuple(sorted(raw_state_dict.keys())),
    }


@dataclass
class ScorePretrainingConfig:
    image_channels: int = 3
    image_size: int = 64
    base_channels: int = 64
    num_res_blocks: int = 2
    attention_downsample_factor: int = 4
    use_output_bias: bool = True
    output_bias_variance: float = 1e-3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    epsilon_min: float = 1e-4
    epsilon_max: float = 5e-2
    score_target: Literal["noise", "mean"] = "noise"
    preview_samples: int = 4
    preview_num_epsilons: int = 5


class ScorePretrainingModule(L.LightningModule):
    def __init__(self, config: ScorePretrainingConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.score_target not in {"noise", "mean"}:
            raise ValueError(
                "score_target must be one of {'noise', 'mean'}, "
                f"got {self.config.score_target}"
            )
        self.save_hyperparameters(asdict(config))
        self.network = ScoreNetwork(
            image_size=config.image_size,
            in_channels=config.image_channels,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_downsample_factor=config.attention_downsample_factor,
            use_output_bias=config.use_output_bias,
            output_bias_variance=config.output_bias_variance,
        )
        self.example_input_array = (
            torch.randn(2, config.image_channels, config.image_size, config.image_size),
            torch.full((2,), 1e-2),
        )

    def forward(self, image: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return self.network(image, epsilon)

    def sample_epsilon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps_min = torch.tensor(self.config.epsilon_min, device=device).log()
        eps_max = torch.tensor(self.config.epsilon_max, device=device).log()
        return torch.exp(torch.rand(batch_size, device=device) * (eps_max - eps_min) + eps_min)

    def predict_noise(self, noisy_images: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(noisy_images, epsilon)
        return prediction_to_noise(
            prediction=prediction,
            noisy_images=noisy_images,
            epsilon=epsilon,
            score_target=self.config.score_target,
        )

    def predict_denoised(self, noisy_images: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(noisy_images, epsilon)
        return prediction_to_denoised(
            prediction=prediction,
            noisy_images=noisy_images,
            epsilon=epsilon,
            score_target=self.config.score_target,
        )

    def compute_score_loss(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = images.shape[0]
        epsilon = self.sample_epsilon(batch_size, images.device)
        noise = torch.randn_like(images)
        noisy_images = images + epsilon.sqrt()[:, None, None, None] * noise
        prediction = self.forward(noisy_images, epsilon)
        target = noise if self.config.score_target == "noise" else images
        loss = (prediction - target).square().mean()
        predicted_noise = prediction_to_noise(
            prediction=prediction,
            noisy_images=noisy_images,
            epsilon=epsilon,
            score_target=self.config.score_target,
        )
        denoised = prediction_to_denoised(
            prediction=prediction,
            noisy_images=noisy_images,
            epsilon=epsilon,
            score_target=self.config.score_target,
        )
        metrics = {
            "score_matching_loss": loss.detach(),
            "epsilon_mean": epsilon.mean().detach(),
            "target_norm": target.square().mean().sqrt().detach(),
            "predicted_noise_norm": predicted_noise.square().mean().sqrt().detach(),
            "denoised_mse": (denoised - images).square().mean().detach(),
        }
        return loss, metrics

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        loss, metrics = self.compute_score_loss(images)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=images.shape[0])
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, prog_bar=False, batch_size=images.shape[0])
        return loss

    def _denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        datamodule = getattr(self.trainer, "datamodule", None)
        stats = getattr(datamodule, "stats", None)
        return restore_image_range(image, stats=stats)

    def _build_preview_canvas(self, cell_images: list[list[torch.Tensor]]) -> np.ndarray:
        rows = len(cell_images)
        cols = len(cell_images[0])
        _, height, width = cell_images[0][0].shape
        gap = 2
        canvas = np.ones(
            (
                rows * height + (rows - 1) * gap,
                cols * width + (cols - 1) * gap,
                3,
            ),
            dtype=np.uint8,
        ) * 255

        for row_idx, row in enumerate(cell_images):
            for col_idx, image in enumerate(row):
                y0 = row_idx * (height + gap)
                x0 = col_idx * (width + gap)
                image_np = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255).byte().cpu().numpy()
                canvas[y0 : y0 + height, x0 : x0 + width] = image_np
        return canvas

    def _preview_epsilon_values(
        self,
        num_values: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_values <= 1:
            epsilon_value = (self.config.epsilon_min * self.config.epsilon_max) ** 0.5
            return torch.full((1,), epsilon_value, device=device, dtype=dtype)
        log_eps = torch.linspace(
            np.log(self.config.epsilon_min),
            np.log(self.config.epsilon_max),
            steps=num_values,
            device=device,
            dtype=dtype,
        )
        return log_eps.exp()

    def _visualize_signed_field(self, field: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        image = 0.5 + 0.5 * (field / scale.clamp_min(1e-6))
        return image.clamp(0.0, 1.0)

    def _log_denoising_examples(self) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return

        num_samples = min(self.config.preview_samples, len(val_dataset))
        clean_images = torch.stack([val_dataset[idx]["image"] for idx in range(num_samples)], dim=0).to(self.device)
        epsilon = self._preview_epsilon_values(1, clean_images.device, clean_images.dtype).expand(num_samples)
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + epsilon.sqrt()[:, None, None, None] * noise

        with torch.no_grad():
            prediction = self.forward(noisy_images, epsilon)
            denoised = prediction_to_denoised(
                prediction=prediction,
                noisy_images=noisy_images,
                epsilon=epsilon,
                score_target=self.config.score_target,
            )
            predicted_noise = prediction_to_noise(
                prediction=prediction,
                noisy_images=noisy_images,
                epsilon=epsilon,
                score_target=self.config.score_target,
            )

        clean_display = self._denormalize_image(clean_images).clamp(0.0, 1.0)
        noisy_display = self._denormalize_image(noisy_images).clamp(0.0, 1.0)
        denoised_display = self._denormalize_image(denoised).clamp(0.0, 1.0)
        residual = clean_display - denoised_display
        noise_scale = torch.stack([noise, predicted_noise], dim=0).square().mean(dim=(0, 2, 3, 4)).sqrt().max()
        residual_scale = residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        rows = [
            [image.cpu() for image in clean_display],
            [image.cpu() for image in noisy_display],
            [image.cpu() for image in denoised_display],
            [self._visualize_signed_field(image, residual_scale).cpu() for image in residual],
            [self._visualize_signed_field(image, noise_scale).cpu() for image in predicted_noise],
        ]
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean/noisy/denoised/clean_minus_denoised/predicted_noise, "
            f"cols=validation samples 0..{num_samples - 1}, "
            f"epsilon={epsilon[0].item():.4g}, residual_scale={residual_scale.item():.4g}, "
            f"score_target={self.config.score_target}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/denoising_examples": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_denoising_epsilon_sweep(self) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return

        clean_image = val_dataset[0]["image"].unsqueeze(0).to(self.device)
        epsilon = self._preview_epsilon_values(
            self.config.preview_num_epsilons,
            clean_image.device,
            clean_image.dtype,
        )
        repeated_clean = clean_image.expand(epsilon.shape[0], -1, -1, -1)
        base_noise = torch.randn_like(clean_image).expand_as(repeated_clean)
        noisy_images = repeated_clean + epsilon.sqrt()[:, None, None, None] * base_noise

        with torch.no_grad():
            prediction = self.forward(noisy_images, epsilon)
            denoised = prediction_to_denoised(
                prediction=prediction,
                noisy_images=noisy_images,
                epsilon=epsilon,
                score_target=self.config.score_target,
            )
            predicted_noise = prediction_to_noise(
                prediction=prediction,
                noisy_images=noisy_images,
                epsilon=epsilon,
                score_target=self.config.score_target,
            )

        clean_display = self._denormalize_image(repeated_clean).clamp(0.0, 1.0)
        noisy_display = self._denormalize_image(noisy_images).clamp(0.0, 1.0)
        denoised_display = self._denormalize_image(denoised).clamp(0.0, 1.0)
        residual = clean_display - denoised_display
        noise_scale = torch.stack([base_noise, predicted_noise], dim=0).square().mean(dim=(0, 2, 3, 4)).sqrt().max()
        residual_scale = residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        base_display = clean_display[0].cpu()
        rows = [
            [base_display.clone() for _ in range(epsilon.shape[0])],
            [image.cpu() for image in noisy_display],
            [image.cpu() for image in denoised_display],
            [self._visualize_signed_field(image, residual_scale).cpu() for image in residual],
            [self._visualize_signed_field(image, noise_scale).cpu() for image in predicted_noise],
        ]
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean/noisy/denoised/clean_minus_denoised/predicted_noise, "
            "cols=epsilon sweep for validation sample 0, "
            f"epsilons={[round(value.item(), 6) for value in epsilon]}, "
            f"residual_scale={residual_scale.item():.4g}, "
            f"score_target={self.config.score_target}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/denoising_by_epsilon": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._log_denoising_examples()
        self._log_denoising_epsilon_sweep()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.trainer.max_epochs, 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
