from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch

from metric_matching.data import restore_image_range
from metric_matching.models import MetricBasisNetwork, ScoreNetwork
from metric_matching.score_module import load_score_network_checkpoint, read_score_checkpoint_config


@dataclass
class AtlasMetricConfig:
    image_channels: int = 3
    image_size: int = 64
    rank: int = 32
    base_channels: int = 64
    num_res_blocks: int = 2
    attention_downsample_factor: int = 4
    use_output_bias: bool = True
    output_bias_variance: float = 1e-3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    denoiser_training_mode: Literal["joint", "pretrained_frozen"] = "joint"
    pretrained_denoiser_checkpoint: str | None = None
    epsilon_min: float = 1e-4
    epsilon_max: float = 5e-2
    scale_input: bool = False
    epsilon_input_mode: str = "log_clamp"
    preview_samples: int = 4
    std_atol: float = 1e-2
    std_rtol: float = 1e-2
    log_var_ema_decay: float = 0.9
    projection_mse_term_weight: float = 1.0


class AtlasMetricModule(L.LightningModule):
    def __init__(self, config: AtlasMetricConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.denoiser_training_mode not in {"joint", "pretrained_frozen"}:
            raise ValueError(
                "denoiser_training_mode must be one of {'joint', 'pretrained_frozen'}, "
                f"got {self.config.denoiser_training_mode}"
            )
        if self.config.denoiser_training_mode == "pretrained_frozen" and self.config.pretrained_denoiser_checkpoint is None:
            raise ValueError(
                "pretrained_denoiser_checkpoint is required when denoiser_training_mode='pretrained_frozen'."
            )
        self.save_hyperparameters(asdict(config))
        self.loaded_denoiser_checkpoint_path: str | None = None
        self.loaded_denoiser_scale_input: bool | None = None
        self.loaded_denoiser_epsilon_input_mode: str | None = None

        denoiser_scale_input = config.scale_input
        denoiser_epsilon_input_mode = config.epsilon_input_mode
        if self.uses_frozen_denoiser:
            checkpoint_config = read_score_checkpoint_config(Path(self.config.pretrained_denoiser_checkpoint))
            denoiser_scale_input = bool(checkpoint_config["scale_input"])
            denoiser_epsilon_input_mode = str(checkpoint_config["epsilon_input_mode"])

        self.denoiser = ScoreNetwork(
            image_size=config.image_size,
            in_channels=config.image_channels,
            data_channels=config.image_channels,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_downsample_factor=config.attention_downsample_factor,
            use_output_bias=config.use_output_bias,
            output_bias_variance=config.output_bias_variance,
            scale_input=denoiser_scale_input,
            epsilon_input_mode=denoiser_epsilon_input_mode,
        )
        if self.uses_frozen_denoiser:
            checkpoint_metadata = load_score_network_checkpoint(
                self.denoiser,
                checkpoint_path=Path(self.config.pretrained_denoiser_checkpoint),
            )
            self.loaded_denoiser_checkpoint_path = str(checkpoint_metadata["checkpoint_path"])
            self.loaded_denoiser_scale_input = bool(checkpoint_metadata["scale_input"])
            self.loaded_denoiser_epsilon_input_mode = str(checkpoint_metadata["epsilon_input_mode"])
            self.denoiser.requires_grad_(False)
            self.denoiser.eval()

        self.projector = MetricBasisNetwork(
            image_size=config.image_size,
            in_channels=config.image_channels,
            data_channels=config.image_channels,
            rank=config.rank,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_downsample_factor=config.attention_downsample_factor,
            use_output_bias=config.use_output_bias,
            output_bias_variance=config.output_bias_variance,
            epsilon_input_mode=config.epsilon_input_mode,
        )

        self.enhancer = ScoreNetwork(
            image_size=config.image_size,
            in_channels=2*config.image_channels,
            data_channels=config.image_channels,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_downsample_factor=config.attention_downsample_factor,
            use_output_bias=config.use_output_bias,
            output_bias_variance=config.output_bias_variance,
            epsilon_input_mode=config.epsilon_input_mode,
        )

        self.register_buffer("projection_log_var", torch.tensor(0.0), persistent=True)
        self.register_buffer("refinement_log_var", torch.tensor(0.0), persistent=True)
        self.register_buffer("log_var_ema_initialized", torch.tensor(False, dtype=torch.bool), persistent=True)

        self.example_input_array = (
            torch.randn(2, config.image_channels, config.image_size, config.image_size),
            torch.full((2,), 1e-2),
        )

    @property
    def uses_frozen_denoiser(self) -> bool:
        return self.config.denoiser_training_mode == "pretrained_frozen"

    def train(self, mode: bool = True):
        super().train(mode)
        if self.uses_frozen_denoiser:
            self.denoiser.eval()
        return self

    def forward(
        self, 
        images: torch.Tensor, 
        epsilon: torch.Tensor,
        local_coord: torch.Tensor | None = None,
    ) -> torch.Tensor:
        var_basis = self.projector(images, epsilon)
        var_basis = var_basis * epsilon.sqrt()[:, None, None, None, None] * self.projection_log_var.div(2).exp()
        
        if local_coord is None:
            local_coord = torch.randn(images.shape[0], var_basis.shape[1], device=images.device, dtype=images.dtype)
        projected_images = images + torch.einsum("bm,bmchw->bchw", local_coord, var_basis)
        enhancer_input = self._build_enhancer_input(images, projected_images)
        enhanced_images = self.enhancer(enhancer_input, epsilon)
        return enhanced_images

    def sample_epsilon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps_min = torch.tensor(self.config.epsilon_min, device=device).log()
        eps_max = torch.tensor(self.config.epsilon_max, device=device).log()
        return torch.exp(torch.rand(batch_size, device=device) * (eps_max - eps_min) + eps_min)

    def _build_enhancer_input(
        self, 
        denoised_images: torch.Tensor,
        projected_images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([denoised_images, projected_images], dim=1)

    def _generate_noise(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor,
        white_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if white_noise is None:
            white_noise = torch.randn_like(images)
        perturbation = epsilon.sqrt()[:, None, None, None] * white_noise
        noisy_images = images + perturbation
        return noisy_images, {
            "perturbation": perturbation,
            "white_noise": white_noise,
        }
    
    def _denoise_images(
        self,
        noisy_images: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        denoised_images = self.denoiser(noisy_images, epsilon)
        return denoised_images

    def _project_images(
        self,
        images: torch.Tensor,
        denoised_images: torch.Tensor,
        epsilon: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        var_basis = self.projector(denoised_images, epsilon)

        var_basis_flat = var_basis.flatten(start_dim=2)
        _, std, basis_flat = torch.linalg.svd(var_basis_flat, full_matrices=False)
        threshold = torch.maximum(self.config.std_atol * torch.ones_like(std[:,0]), self.config.std_rtol * std[:,0])
        threshold = threshold.unsqueeze(1)
        mask = std > threshold
        basis = basis_flat.view_as(var_basis)

        diff = images - denoised_images
        latent = torch.einsum("bnchw,bchw->bn", basis, diff)
        latent = latent * mask.float()
        projected_images = denoised_images + torch.einsum("bm,bmchw->bchw", latent, basis)

        return projected_images, {
            "latent": latent,
            "var_basis": var_basis,
            "std": std,
            "threshold": threshold,
            "mask": mask,
        }
    
    def _enhance_images(
        self,
        denoised_images: torch.Tensor,
        projected_images: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        enhancer_input = self._build_enhancer_input(denoised_images, projected_images)
        enhanced_images = self.enhancer(enhancer_input, epsilon)
        return enhanced_images
    
    def _compute_nll(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        std = aux["std"]
        mask = aux["mask"]
        threshold = aux["threshold"]
        latent = aux["latent"]
        prc = std.clamp_min(threshold).reciprocal() * mask.float()
        basis_log_var = 2 * std.clamp_min(threshold).log() * mask.float()

        data_dim = self.config.image_channels * self.config.image_size * self.config.image_size
        tangent_dim = mask.sum(dim=1)
        tangent_dim_float = tangent_dim.to(dtype=images.dtype)
        normal_dim = data_dim - tangent_dim
        normal_dim_float = normal_dim.to(dtype=images.dtype)

        denoised_images = aux["denoised_images"]
        denoising_mse_term = (images - denoised_images).square().sum(dim=(1, 2, 3)).div(epsilon)
        denoising_mse_term = denoising_mse_term / data_dim

        projected_images = aux["projected_images"]
        projection_mse_term = (images - projected_images).square().sum(dim=(1, 2, 3)).div(epsilon)
        projection_mse_term = projection_mse_term / data_dim
        weighted_projection_mse_term = projection_mse_term * self.config.projection_mse_term_weight

        projection_trace_coeff = latent.mul(prc).square().sum(dim=1).div(epsilon)
        enhanced_images = aux["enhanced_images"]
        refinement_trace_coeff = (enhanced_images - images).square().sum(dim=(1, 2, 3)).div(epsilon)

        tiny = torch.finfo(images.dtype).tiny
        projection_log_var = (
            projection_trace_coeff.mean().clamp_min(tiny).log()
            - tangent_dim_float.mean().clamp_min(tiny).log()
        )
        refinement_log_var = (
            refinement_trace_coeff.mean().clamp_min(tiny).log()
            - normal_dim_float.mean().clamp_min(tiny).log()
        )

        projection_log_var = torch.where(
            tangent_dim_float.mean() > 0,
            projection_log_var,
            torch.zeros_like(projection_log_var),
        )
        refinement_log_var = torch.where(
            normal_dim_float.mean() > 0,
            refinement_log_var,
            torch.zeros_like(refinement_log_var),
        )

        projection_trace_term = projection_trace_coeff / projection_log_var.exp()
        projection_logdet_term = basis_log_var.sum(dim=1) + tangent_dim_float * (torch.log(epsilon) + projection_log_var)
        projection_nll = 0.5 * projection_trace_term + 0.5 * projection_logdet_term
        projection_nll = projection_nll / data_dim

        refinement_trace_term = refinement_trace_coeff / refinement_log_var.exp()
        refinement_logdet_term = normal_dim_float * (torch.log(epsilon) + refinement_log_var)
        refinement_nll = 0.5 * refinement_trace_term + 0.5 * refinement_logdet_term
        refinement_nll = refinement_nll / data_dim

        with torch.no_grad():
            if bool(self.log_var_ema_initialized):
                ema_decay = torch.as_tensor(
                    self.config.log_var_ema_decay,
                    device=self.projection_log_var.device,
                    dtype=self.projection_log_var.dtype,
                )
                one_minus_decay = 1.0 - ema_decay
                self.projection_log_var.mul_(ema_decay).add_(projection_log_var.detach().to(self.projection_log_var.dtype) * one_minus_decay)
                self.refinement_log_var.mul_(ema_decay).add_(refinement_log_var.detach().to(self.refinement_log_var.dtype) * one_minus_decay)
            else:
                self.projection_log_var.copy_(projection_log_var.detach())
                self.refinement_log_var.copy_(refinement_log_var.detach())
                self.log_var_ema_initialized.fill_(True)

        nll = denoising_mse_term + weighted_projection_mse_term + projection_nll + refinement_nll
        return nll, {
            **aux,
            "denoising_mse_term": denoising_mse_term,
            "projection_mse_term": projection_mse_term,
            "weighted_projection_mse_term": weighted_projection_mse_term,
            "projection_nll": projection_nll,
            "refinement_nll": refinement_nll,
            "projection_log_var": projection_log_var,
            "refinement_log_var": refinement_log_var,
        }

    def _compute_outputs(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor,
        white_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        noisy_images, generated = self._generate_noise(
            images,
            epsilon,
            white_noise=white_noise,
        )
        denoised_images = self._denoise_images(noisy_images, epsilon)
        projected_images, projected = self._project_images(images, denoised_images.detach(), epsilon)
        enhanced_images = self._enhance_images(denoised_images.detach(), projected_images, epsilon)
        
        return {
            "epsilon": epsilon,
            "clean_images": images,
            "noisy_images": noisy_images,
            "denoised_images": denoised_images,
            "projected_images": projected_images,
            "enhanced_images": enhanced_images,
            "perturbation": generated["perturbation"],
            "white_noise": generated["white_noise"],
            "basis": projected["var_basis"],
            "latent": projected["latent"],
            "std": projected["std"],
            "mask": projected["mask"],
            "threshold": projected["threshold"],
        }

    def _run_step(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor | None = None,
        white_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if epsilon is None:
            epsilon = self.sample_epsilon(images.shape[0], images.device).to(dtype=images.dtype)
        if white_noise is None:
            white_noise = torch.randn_like(images)
        aux = self._compute_outputs(images, epsilon, white_noise)
        nll, aux = self._compute_nll(images, epsilon, aux)

        metrics = {
            "nll": nll.mean().detach(),
            "denoising_mse_term": aux["denoising_mse_term"].mean().detach(),
            "projection_mse_term": aux["projection_mse_term"].mean().detach(),
            "projection_nll": aux["projection_nll"].mean().detach(),
            "refinement_nll": aux["refinement_nll"].mean().detach(),
            "white_noise_rms": aux["white_noise"].square().mean().sqrt().detach(),
            "perturbation_rms": aux["perturbation"].square().mean().sqrt().detach(),
            "projection_log_var": aux["projection_log_var"].detach(),
            "refinement_log_var": aux["refinement_log_var"].detach(),
            "tangent_dim": aux["mask"].float().sum(dim=1).mean().detach(),
            "latent_var": aux["std"].square().sum(dim=1).mean().div(aux["projection_log_var"]).detach(),
        }
        return nll.mean(), metrics

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        loss, metrics = self._run_step(images)
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

    @staticmethod
    def _warmup_decay_lr_lambda(
        alpha: float,
        warmup_steps: int,
        scale_steps: int,
    ) -> Callable[[int], float]:
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0:
                warmup = min(float(step + 1) / float(warmup_steps), 1.0)
            else:
                warmup = 1.0
            effective_scale_steps = max(scale_steps, 1)
            scaled_step = float(step) / float(effective_scale_steps)
            return warmup * float((scaled_step + 1.0) ** (-alpha))

        return lr_lambda

    def _visualize_signed_field(self, field: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        image = 0.5 + 0.5 * (field / scale.clamp_min(1e-6))
        return image.clamp(0.0, 1.0)

    def _top_metric_singular_vectors(self, basis_fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_basis = basis_fields.flatten(start_dim=2)
        _, singular_values, vh = torch.linalg.svd(flat_basis, full_matrices=False)
        singular_vectors = vh.view(vh.shape[0], vh.shape[1], *basis_fields.shape[2:])
        return singular_vectors, singular_values

    def _build_sample_preview(self) -> dict[str, torch.Tensor] | None:
        if self.trainer is None or self.trainer.sanity_checking:
            return None
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return None

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return None

        num_samples = min(self.config.preview_samples, len(val_dataset))
        clean_images = torch.stack([val_dataset[idx]["image"] for idx in range(num_samples)], dim=0).to(self.device)
        epsilon = self._preview_epsilon_values(1, clean_images.device, clean_images.dtype).expand(num_samples)
        white_noise = torch.randn_like(clean_images)

        with torch.no_grad():
            preview = self._compute_outputs(clean_images, epsilon, white_noise)
            preview["basis_singular_vectors"], preview["basis_singular_values"] = self._top_metric_singular_vectors(
                preview["basis"]
            )
        return preview

    def _build_forward_sample_preview(self) -> dict[str, torch.Tensor] | None:
        if self.trainer is None or self.trainer.sanity_checking:
            return None
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return None

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return None

        num_samples = min(self.config.preview_samples, len(val_dataset))
        clean_images = torch.stack([val_dataset[idx]["image"] for idx in range(num_samples)], dim=0).to(self.device)
        epsilon = self._preview_epsilon_values(1, clean_images.device, clean_images.dtype).expand(num_samples)
        num_coords = min(8, self.config.rank)
        local_coord = torch.randn(num_coords, num_samples, self.config.rank, device=self.device, dtype=clean_images.dtype)

        with torch.no_grad():
            forward_images = torch.stack(
                [self(clean_images, epsilon, local_coord=coords) for coords in local_coord],
                dim=0,
            )

        return {
            "clean_images": clean_images,
            "forward_images": forward_images,
            "epsilon": epsilon,
            "local_coord": local_coord,
            "num_coords": num_coords,
        }

    def _build_epsilon_preview(self) -> dict[str, torch.Tensor] | None:
        if self.trainer is None or self.trainer.sanity_checking:
            return None
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return None

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return None

        clean_image = val_dataset[0]["image"].unsqueeze(0).to(self.device)
        epsilon = self._preview_epsilon_values(
            max(1, self.config.preview_samples),
            clean_image.device,
            clean_image.dtype,
        )
        repeated_clean = clean_image.expand(epsilon.shape[0], -1, -1, -1)
        white_noise = torch.randn_like(repeated_clean)

        with torch.no_grad():
            preview = self._compute_outputs(repeated_clean, epsilon, white_noise)
            preview["basis_singular_vectors"], preview["basis_singular_values"] = self._top_metric_singular_vectors(
                preview["basis"]
            )
        return preview

    def _build_forward_epsilon_preview(self) -> dict[str, torch.Tensor] | None:
        if self.trainer is None or self.trainer.sanity_checking:
            return None
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return None

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return None

        clean_image = val_dataset[0]["image"].unsqueeze(0).to(self.device)
        epsilon = self._preview_epsilon_values(
            max(1, self.config.preview_samples),
            clean_image.device,
            clean_image.dtype,
        )
        repeated_clean = clean_image.expand(epsilon.shape[0], -1, -1, -1)
        num_coords = min(8, self.config.rank)
        base_local_coord = torch.randn(num_coords, 1, self.config.rank, device=self.device, dtype=clean_image.dtype)
        local_coord = base_local_coord.expand(-1, epsilon.shape[0], -1)

        with torch.no_grad():
            forward_images = torch.stack(
                [self(repeated_clean, epsilon, local_coord=coords) for coords in local_coord],
                dim=0,
            )

        return {
            "clean_images": repeated_clean,
            "forward_images": forward_images,
            "epsilon": epsilon,
            "local_coord": local_coord,
            "num_coords": num_coords,
        }

    def _log_image_comparison_grid(self, preview: dict[str, torch.Tensor]) -> None:
        epsilon = preview["epsilon"]
        clean_display = self._denormalize_image(preview["clean_images"]).clamp(0.0, 1.0)
        noisy_display = self._denormalize_image(preview["noisy_images"]).clamp(0.0, 1.0)
        denoised_display = self._denormalize_image(preview["denoised_images"]).clamp(0.0, 1.0)
        projected_display = self._denormalize_image(preview["projected_images"]).clamp(0.0, 1.0)
        enhanced_display = self._denormalize_image(preview["enhanced_images"]).clamp(0.0, 1.0)
        noisy_residual = noisy_display - clean_display
        denoised_residual = denoised_display - clean_display
        projected_residual = projected_display - clean_display
        enhanced_residual = enhanced_display - clean_display
        noisy_residual_scale = noisy_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        denoised_residual_scale = denoised_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        projected_residual_scale = projected_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        enhanced_residual_scale = enhanced_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)

        rows = [
            [image.cpu() for image in clean_display],
            [image.cpu() for image in noisy_display],
            [image.cpu() for image in denoised_display],
            [image.cpu() for image in projected_display],
            [image.cpu() for image in enhanced_display],
            [self._visualize_signed_field(image, noisy_residual_scale).cpu() for image in noisy_residual],
            [self._visualize_signed_field(image, denoised_residual_scale).cpu() for image in denoised_residual],
            [self._visualize_signed_field(image, projected_residual_scale).cpu() for image in projected_residual],
            [self._visualize_signed_field(image, enhanced_residual_scale).cpu() for image in enhanced_residual],
        ]
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean/noisy/denoised/projected/enhanced/noisy_minus_clean/denoised_minus_clean/projected_minus_clean/enhanced_minus_clean, "
            f"cols=validation samples 0..{preview['clean_images'].shape[0] - 1}, "
            f"epsilon={epsilon[0].item():.4g}, "
            f"noisy_residual_scale={noisy_residual_scale.item():.4g}, "
            f"denoised_residual_scale={denoised_residual_scale.item():.4g}, "
            f"projected_residual_scale={projected_residual_scale.item():.4g}, "
            f"enhanced_residual_scale={enhanced_residual_scale.item():.4g}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/examples": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_image_epsilon_grid(self, preview: dict[str, torch.Tensor]) -> None:
        epsilon = preview["epsilon"]
        clean_display = self._denormalize_image(preview["clean_images"]).clamp(0.0, 1.0)
        noisy_display = self._denormalize_image(preview["noisy_images"]).clamp(0.0, 1.0)
        denoised_display = self._denormalize_image(preview["denoised_images"]).clamp(0.0, 1.0)
        projected_display = self._denormalize_image(preview["projected_images"]).clamp(0.0, 1.0)
        enhanced_display = self._denormalize_image(preview["enhanced_images"]).clamp(0.0, 1.0)
        noisy_residual = (noisy_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        denoised_residual = (denoised_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        projected_residual = (projected_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        enhanced_residual = (enhanced_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        noisy_residual_scale = noisy_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        denoised_residual_scale = denoised_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        projected_residual_scale = projected_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        enhanced_residual_scale = enhanced_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        base_display = clean_display[0].cpu()

        rows = [
            [base_display.clone() for _ in range(epsilon.shape[0])],
            [image.cpu() for image in noisy_display],
            [image.cpu() for image in denoised_display],
            [image.cpu() for image in projected_display],
            [image.cpu() for image in enhanced_display],
            [self._visualize_signed_field(image, noisy_residual_scale).cpu() for image in noisy_residual],
            [self._visualize_signed_field(image, denoised_residual_scale).cpu() for image in denoised_residual],
            [self._visualize_signed_field(image, projected_residual_scale).cpu() for image in projected_residual],
            [self._visualize_signed_field(image, enhanced_residual_scale).cpu() for image in enhanced_residual],
        ]
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean/noisy/denoised/projected/enhanced/noisy_minus_clean/denoised_minus_clean/projected_minus_clean/enhanced_minus_clean, "
            "cols=epsilon sweep for validation sample 0, "
            f"epsilons={[round(value.item(), 6) for value in epsilon]}, "
            f"noisy_residual_scale={noisy_residual_scale.item():.4g}, "
            f"denoised_residual_scale={denoised_residual_scale.item():.4g}, "
            f"projected_residual_scale={projected_residual_scale.item():.4g}, "
            f"enhanced_residual_scale={enhanced_residual_scale.item():.4g}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/by_epsilon": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_forward_sample_grid(self, preview: dict[str, torch.Tensor]) -> None:
        epsilon = preview["epsilon"]
        clean_display = self._denormalize_image(preview["clean_images"]).clamp(0.0, 1.0)
        forward_display = self._denormalize_image(preview["forward_images"].flatten(0, 1)).clamp(0.0, 1.0).view_as(preview["forward_images"])
        forward_residual = forward_display - clean_display.unsqueeze(0)
        forward_residual_scale = (
            forward_residual.square().mean(dim=(2, 3, 4)).sqrt().max().mul(3.0).clamp_min(1e-6)
        )

        rows: list[list[torch.Tensor]] = []
        rows.append([image.cpu() for image in clean_display])
        for coord_idx in range(preview["num_coords"]):
            rows.append([image.cpu() for image in forward_display[coord_idx]])
            rows.append(
                [
                    self._visualize_signed_field(image, forward_residual_scale).cpu()
                    for image in forward_residual[coord_idx]
                ]
            )
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean then per-coord pairs of forward/forward_minus_clean, "
            f"cols=validation samples 0..{preview['clean_images'].shape[0] - 1}, "
            f"epsilon={epsilon[0].item():.4g}, "
            f"num_coords={preview['num_coords']}, "
            f"forward_residual_scale={forward_residual_scale.item():.4g}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/forward_examples": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_forward_epsilon_grid(self, preview: dict[str, torch.Tensor]) -> None:
        epsilon = preview["epsilon"]
        clean_display = self._denormalize_image(preview["clean_images"]).clamp(0.0, 1.0)
        forward_display = self._denormalize_image(preview["forward_images"].flatten(0, 1)).clamp(0.0, 1.0).view_as(preview["forward_images"])
        forward_residual = (forward_display - clean_display.unsqueeze(0)) / epsilon.sqrt()[None, :, None, None, None]
        forward_residual_scale = (
            forward_residual.square().mean(dim=(2, 3, 4)).sqrt().max().mul(3.0).clamp_min(1e-6)
        )
        base_display = clean_display[0].cpu()

        rows: list[list[torch.Tensor]] = []
        rows.append([base_display.clone() for _ in range(epsilon.shape[0])])
        for coord_idx in range(preview["num_coords"]):
            rows.append([image.cpu() for image in forward_display[coord_idx]])
            rows.append(
                [
                    self._visualize_signed_field(image, forward_residual_scale).cpu()
                    for image in forward_residual[coord_idx]
                ]
            )
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean then per-coord pairs of forward/forward_minus_clean_over_sqrt_epsilon, "
            "cols=epsilon sweep for validation sample 0, "
            f"epsilons={[round(value.item(), 6) for value in epsilon]}, "
            f"num_coords={preview['num_coords']}, "
            f"forward_residual_scale={forward_residual_scale.item():.4g}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/forward_by_epsilon": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_basis_comparison_grid(
        self,
        singular_vectors: torch.Tensor,
        singular_values: torch.Tensor,
        clean_images: torch.Tensor,
        epsilon: torch.Tensor,
        log_key: str,
        basis_label: str,
    ) -> None:
        num_samples = clean_images.shape[0]
        num_fields = min(8, singular_vectors.shape[1])
        rows: list[list[torch.Tensor]] = []
        rows.append([self._denormalize_image(image).clamp(0.0, 1.0).cpu() for image in clean_images])

        displayed_singular_values = []
        for field_idx in range(num_fields):
            row_fields = singular_vectors[:, field_idx]
            row_scale = row_fields.square().mean(dim=(1, 2, 3)).sqrt().max().clamp_min(1e-6)
            row_images = [
                self._visualize_signed_field(row_fields[sample_idx], row_scale).cpu()
                for sample_idx in range(num_samples)
            ]
            rows.append(row_images)
            displayed_singular_values.append(singular_values[:, field_idx].mean().item())

        canvas = self._build_preview_canvas(rows)
        top_row_label = "denoised images"
        caption = (
            f"top row={top_row_label}, lower rows=top singular vectors of {basis_label} 0..{num_fields - 1}, "
            f"cols=validation samples 0..{num_samples - 1}, epsilon={epsilon[0].item():.4g}, "
            f"mean_singular_values={[round(v, 4) for v in displayed_singular_values]}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    log_key: wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_basis_epsilon_grid(
        self,
        singular_vectors: torch.Tensor,
        singular_values: torch.Tensor,
        clean_images: torch.Tensor,
        epsilon: torch.Tensor,
        log_key: str,
        basis_label: str,
    ) -> None:
        num_fields = min(8, singular_vectors.shape[1])
        rows: list[list[torch.Tensor]] = []
        rows.append([
            self._denormalize_image(clean_images[epsilon_idx]).clamp(0.0, 1.0).cpu()
            for epsilon_idx in range(epsilon.shape[0])
        ])

        displayed_singular_values = []
        for field_idx in range(num_fields):
            row_fields = singular_vectors[:, field_idx]
            row_scale = row_fields.square().mean(dim=(1, 2, 3)).sqrt().max().clamp_min(1e-6)
            row_images = [
                self._visualize_signed_field(row_fields[epsilon_idx], row_scale).cpu()
                for epsilon_idx in range(epsilon.shape[0])
            ]
            rows.append(row_images)
            displayed_singular_values.append([round(v.item(), 4) for v in singular_values[:, field_idx]])

        canvas = self._build_preview_canvas(rows)
        top_row_label = "denoised image"
        caption = (
            f"top row={top_row_label}, lower rows=top singular vectors of {basis_label} 0..{num_fields - 1}, "
            f"cols=epsilon sweep for validation sample 0, epsilons={[round(v.item(), 6) for v in epsilon]}, "
            f"singular_values_by_row={displayed_singular_values}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    log_key: wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        epsilon = self.sample_epsilon(images.shape[0], images.device).to(dtype=images.dtype)
        white_noise = torch.randn_like(images)
        loss, metrics = self._run_step(
            images,
            epsilon=epsilon,
            white_noise=white_noise,
        )

        self.log("train/loss", loss, prog_bar=True, batch_size=images.shape[0])
        for name, value in metrics.items():
            self.log(f"train/{name}", value, prog_bar=False, batch_size=images.shape[0])
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        sample_preview = self._build_sample_preview()
        epsilon_preview = self._build_epsilon_preview()
        forward_sample_preview = self._build_forward_sample_preview()
        forward_epsilon_preview = self._build_forward_epsilon_preview()
        if (
            sample_preview is None
            or epsilon_preview is None
            or forward_sample_preview is None
            or forward_epsilon_preview is None
        ):
            return

        self._log_image_comparison_grid(sample_preview)
        self._log_image_epsilon_grid(epsilon_preview)
        self._log_forward_sample_grid(forward_sample_preview)
        self._log_forward_epsilon_grid(forward_epsilon_preview)
        self._log_basis_comparison_grid(
            singular_vectors=sample_preview["basis_singular_vectors"],
            singular_values=sample_preview["basis_singular_values"],
            clean_images=sample_preview["denoised_images"],
            epsilon=sample_preview["epsilon"],
            log_key="val/basis_vectors",
            basis_label="basis around denoised image",
        )
        self._log_basis_epsilon_grid(
            singular_vectors=epsilon_preview["basis_singular_vectors"],
            singular_values=epsilon_preview["basis_singular_values"],
            clean_images=epsilon_preview["denoised_images"],
            epsilon=epsilon_preview["epsilon"],
            log_key="val/basis_vectors_by_epsilon",
            basis_label="basis around denoised image",
        )

    def configure_optimizers(self):
        trainable_parameters: list[torch.nn.Parameter] = []
        if not self.uses_frozen_denoiser:
            trainable_parameters.extend(list(self.denoiser.parameters()))
        trainable_parameters.extend(list(self.projector.parameters()))
        trainable_parameters.extend(list(self.enhancer.parameters()))
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.trainer.max_epochs, 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
