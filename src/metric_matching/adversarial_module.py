from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import lightning as L
import numpy as np
import torch

from metric_matching.data import restore_image_range
from metric_matching.models import MetricFactorNetwork, MetricBasisNetwork, ScoreNetwork


@dataclass
class AdversarialMetricConfig:
    image_channels: int = 3
    image_size: int = 64
    rank: int = 32
    base_channels: int = 64
    num_res_blocks: int = 2
    attention_downsample_factor: int = 4
    use_output_bias: bool = True
    output_bias_variance: float = 1e-3
    denoiser_learning_rate: float = 2e-4
    generator_learning_rate: float = 2e-4
    denoiser_weight_decay: float = 0.0
    generator_weight_decay: float = 0.0
    epsilon_min: float = 1e-4
    epsilon_max: float = 5e-2
    generator_loss_weight: float = 1.0
    covariance_regularization: float = 1e-6
    scale_input: bool = False
    epsilon_input_mode: str = "log_clamp"
    preview_samples: int = 4
    denoiser_lr_alpha: float = 0.0
    generator_lr_alpha: float = 0.0
    denoiser_warmup_steps: int = 0
    generator_warmup_steps: int = 0
    denoiser_lr_scale_steps: int = 1
    generator_lr_scale_steps: int = 1


class AdversarialMetricModule(L.LightningModule):
    def __init__(self, config: AdversarialMetricConfig) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.automatic_optimization = False

        self.projector = MetricFactorNetwork(
            image_size=config.image_size,
            in_channels=config.image_channels,
            data_channels=config.image_channels,
            rank=config.rank,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_downsample_factor=config.attention_downsample_factor,
            use_output_bias=config.use_output_bias,
            output_bias_variance=config.output_bias_variance,
            scale_input=config.scale_input,
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
            scale_input=config.scale_input,
            epsilon_input_mode=config.epsilon_input_mode,
        )

        self.noise_generator = MetricBasisNetwork(
            image_size=config.image_size,
            in_channels=config.image_channels,
            data_channels=config.image_channels,
            rank=config.rank,
            base_channels=config.base_channels,
            num_res_blocks=config.num_res_blocks,
            attention_downsample_factor=config.attention_downsample_factor,
            use_output_bias=config.use_output_bias,
            output_bias_variance=config.output_bias_variance,
            scale_input=config.scale_input,
            epsilon_input_mode=config.epsilon_input_mode,
        )

        self.projector_log_var = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.enhancer_log_var = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.example_input_array = (
            torch.randn(2, config.image_channels, config.image_size, config.image_size),
            torch.full((2,), 1e-2),
        )

    def forward(self, images: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return self.noise_generator(images, epsilon)

    def sample_epsilon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps_min = torch.tensor(self.config.epsilon_min, device=device).log()
        eps_max = torch.tensor(self.config.epsilon_max, device=device).log()
        return torch.exp(torch.rand(batch_size, device=device) * (eps_max - eps_min) + eps_min)

    def _normalize_metric_basis(self, metric_basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        basis_rms = metric_basis.square().mean(dim=(2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        return metric_basis / basis_rms, basis_rms
    
    def _build_enhancer_input(
        self, 
        noisy_images: torch.Tensor,
        projected_images: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([noisy_images, projected_images], dim=1)

    def _generate_noise(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor,
        latent_white_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metric_basis_prediction = self.noise_generator(images, epsilon)
        normalized_basis, basis_rms = self._normalize_metric_basis(metric_basis_prediction)
        if latent_white_noise is None:
            latent_white_noise = torch.randn(
                images.shape[0],
                self.config.rank,
                device=images.device,
                dtype=images.dtype,
            )
        structured_noise = torch.einsum("bk,bkchw->bchw", latent_white_noise, normalized_basis)
        perturbation = epsilon.sqrt()[:, None, None, None] * structured_noise
        noisy_images = images + perturbation
        return noisy_images, {
            "perturbation": perturbation,
            "structured_noise": structured_noise,
            "normalized_basis": normalized_basis,
            "basis_rms_before_norm": basis_rms.squeeze(-1).squeeze(-1).squeeze(-1),
            "latent_white_noise": latent_white_noise,
        }

    def _project_images(
        self,
        images: torch.Tensor,
        noisy_images: torch.Tensor,
        epsilon: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        basis, mean = self.projector(noisy_images, epsilon)
        # basis = basis + noisy_images.unsqueeze(1)
        # mean = mean + noisy_images
        var_basis = basis - mean.unsqueeze(1)
        covariance = torch.einsum("bmchw,bnchw->bmn", var_basis, var_basis)
        eye = torch.eye(
            covariance.shape[-1],
            device=covariance.device,
            dtype=covariance.dtype,
        ).unsqueeze(0)
        covariance = covariance + eye * self.config.covariance_regularization
        covariance_inv = torch.linalg.inv(covariance)

        diff = images - mean
        latent = torch.einsum("bnchw,bchw->bn", var_basis, diff)
        latent = torch.einsum("bmn,bn->bm", covariance_inv, latent)
        projected_images = mean + torch.einsum("bm,bmchw->bchw", latent, var_basis)

        return projected_images, {
            "covariance": covariance,
            "latent": latent,
            "mean": mean,
            "var_basis": var_basis,
        }
    
    def _enhance_images(
        self,
        noisy_images: torch.Tensor,
        projected_images: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        enhancer_input = self._build_enhancer_input(noisy_images, projected_images)
        enhanced_images = self.enhancer(enhancer_input, epsilon)
        return enhanced_images
    
    def _compute_nll(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        data_dim = self.config.image_channels * self.config.image_size * self.config.image_size
        tangent_dim = self.config.rank
        normal_dim = data_dim - tangent_dim

        covariance = aux["covariance"]
        latent = aux["latent"]
        project_trace_term = latent.square().sum(dim=1).div(epsilon * self.projector_log_var.exp())
        project_logdet_term = torch.logdet(covariance) + tangent_dim * (torch.log(epsilon) + self.projector_log_var)
        project_nll = 0.5 * project_trace_term + 0.5 * project_logdet_term

        enhanced_images = aux["enhanced_images"]
        enhance_trace_term = (enhanced_images - images).square().sum(dim=(1, 2, 3)).div(epsilon * self.enhancer_log_var.exp())
        enhance_logdet_term = normal_dim * (torch.log(epsilon) + self.enhancer_log_var)
        enhance_nll = 0.5 * enhance_trace_term + 0.5 * enhance_logdet_term

        nll = (project_nll + enhance_nll) / data_dim
        return nll, {
            **aux,
            "project_nll": project_nll,
            "enhance_nll": enhance_nll,
        }

    def _compute_adversarial_outputs(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor,
        latent_white_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        noisy_images, generated = self._generate_noise(
            images,
            epsilon,
            latent_white_noise=latent_white_noise,
        )
        projected_images, projected = self._project_images(images, noisy_images, epsilon)
        enhanced_images = self._enhance_images(noisy_images, projected_images, epsilon)
        return {
            "epsilon": epsilon,
            "clean_images": images,
            "noisy_images": noisy_images,
            "perturbation": generated["perturbation"],
            "structured_noise": generated["structured_noise"],
            "basis_rms_before_norm": generated["basis_rms_before_norm"],
            "projected_images": projected_images,
            "mean_images": projected["mean"],
            "enhanced_images": enhanced_images,
            "generated_basis": generated["normalized_basis"],
            "projector_basis": projected["var_basis"],
            "covariance": projected["covariance"],
            "latent": projected["latent"],
        }

    def _run_adversarial_round(
        self,
        images: torch.Tensor,
        epsilon: torch.Tensor | None = None,
        latent_white_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if epsilon is None:
            epsilon = self.sample_epsilon(images.shape[0], images.device).to(dtype=images.dtype)
        if latent_white_noise is None:
            latent_white_noise = torch.randn(
                images.shape[0],
                self.config.rank,
                device=images.device,
                dtype=images.dtype,
            )
        aux = self._compute_adversarial_outputs(images, epsilon, latent_white_noise)
        nll, aux = self._compute_nll(images, epsilon, aux)
        generated_basis = aux["generated_basis"]
        
        metrics = {
            "nll": nll.mean().detach(),
            "generator_basis_rms_before_norm": aux["basis_rms_before_norm"].mean().detach(),
            "generator_basis_rms_after_norm": generated_basis
            .square()
            .mean(dim=(2, 3, 4))
            .sqrt()
            .mean()
            .detach(),
            "structured_noise_rms": aux["structured_noise"].square().mean().sqrt().detach(),
            "perturbation_rms": aux["perturbation"].square().mean().sqrt().detach(),
            "projector_log_var": self.projector_log_var.detach(),
            "enhancer_log_var": self.enhancer_log_var.detach(),
        }
        return nll.mean(), metrics

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        loss, metrics = self._run_adversarial_round(images)
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
        latent_white_noise = torch.randn(
            num_samples,
            self.config.rank,
            device=self.device,
            dtype=clean_images.dtype,
        )

        with torch.no_grad():
            preview = self._compute_adversarial_outputs(clean_images, epsilon, latent_white_noise)
            preview["generated_singular_vectors"], preview["generated_singular_values"] = self._top_metric_singular_vectors(
                preview["generated_basis"]
            )
            preview["projector_singular_vectors"], preview["projector_singular_values"] = self._top_metric_singular_vectors(
                preview["projector_basis"]
            )
        return preview

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
        latent_white_noise = torch.randn(
            1,
            self.config.rank,
            device=self.device,
            dtype=clean_image.dtype,
        ).expand(epsilon.shape[0], -1)

        with torch.no_grad():
            preview = self._compute_adversarial_outputs(repeated_clean, epsilon, latent_white_noise)
            preview["generated_singular_vectors"], preview["generated_singular_values"] = self._top_metric_singular_vectors(
                preview["generated_basis"]
            )
            preview["projector_singular_vectors"], preview["projector_singular_values"] = self._top_metric_singular_vectors(
                preview["projector_basis"]
            )
        return preview

    def _log_image_comparison_grid(self, preview: dict[str, torch.Tensor]) -> None:
        epsilon = preview["epsilon"]
        clean_display = self._denormalize_image(preview["clean_images"]).clamp(0.0, 1.0)
        noisy_display = self._denormalize_image(preview["noisy_images"]).clamp(0.0, 1.0)
        mean_display = self._denormalize_image(preview["mean_images"]).clamp(0.0, 1.0)
        projected_display = self._denormalize_image(preview["projected_images"]).clamp(0.0, 1.0)
        enhanced_display = self._denormalize_image(preview["enhanced_images"]).clamp(0.0, 1.0)
        noisy_residual = noisy_display - clean_display
        mean_residual = mean_display - clean_display
        projected_residual = projected_display - clean_display
        enhanced_residual = enhanced_display - clean_display
        noisy_residual_scale = noisy_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        mean_residual_scale = mean_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        projected_residual_scale = projected_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        enhanced_residual_scale = enhanced_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)

        rows = [
            [image.cpu() for image in clean_display],
            [image.cpu() for image in noisy_display],
            [image.cpu() for image in mean_display],
            [image.cpu() for image in projected_display],
            [image.cpu() for image in enhanced_display],
            [self._visualize_signed_field(image, noisy_residual_scale).cpu() for image in noisy_residual],
            [self._visualize_signed_field(image, mean_residual_scale).cpu() for image in mean_residual],
            [self._visualize_signed_field(image, projected_residual_scale).cpu() for image in projected_residual],
            [self._visualize_signed_field(image, enhanced_residual_scale).cpu() for image in enhanced_residual],
        ]
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean/noisy/mean/projected/enhanced/noisy_minus_clean/mean_minus_clean/projected_minus_clean/enhanced_minus_clean, "
            f"cols=validation samples 0..{preview['clean_images'].shape[0] - 1}, "
            f"epsilon={epsilon[0].item():.4g}, "
            f"noisy_residual_scale={noisy_residual_scale.item():.4g}, "
            f"mean_residual_scale={mean_residual_scale.item():.4g}, "
            f"projected_residual_scale={projected_residual_scale.item():.4g}, "
            f"enhanced_residual_scale={enhanced_residual_scale.item():.4g}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/adversarial_examples": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_image_epsilon_grid(self, preview: dict[str, torch.Tensor]) -> None:
        epsilon = preview["epsilon"]
        clean_display = self._denormalize_image(preview["clean_images"]).clamp(0.0, 1.0)
        noisy_display = self._denormalize_image(preview["noisy_images"]).clamp(0.0, 1.0)
        mean_display = self._denormalize_image(preview["mean_images"]).clamp(0.0, 1.0)
        projected_display = self._denormalize_image(preview["projected_images"]).clamp(0.0, 1.0)
        enhanced_display = self._denormalize_image(preview["enhanced_images"]).clamp(0.0, 1.0)
        noisy_residual = (noisy_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        mean_residual = (mean_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        projected_residual = (projected_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        enhanced_residual = (enhanced_display - clean_display) / epsilon.sqrt()[:, None, None, None]
        noisy_residual_scale = noisy_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        mean_residual_scale = mean_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        projected_residual_scale = projected_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        enhanced_residual_scale = enhanced_residual.square().mean(dim=(1, 2, 3)).sqrt().max().mul(3.0).clamp_min(1e-6)
        base_display = clean_display[0].cpu()

        rows = [
            [base_display.clone() for _ in range(epsilon.shape[0])],
            [image.cpu() for image in noisy_display],
            [image.cpu() for image in mean_display],
            [image.cpu() for image in projected_display],
            [image.cpu() for image in enhanced_display],
            [self._visualize_signed_field(image, noisy_residual_scale).cpu() for image in noisy_residual],
            [self._visualize_signed_field(image, mean_residual_scale).cpu() for image in mean_residual],
            [self._visualize_signed_field(image, projected_residual_scale).cpu() for image in projected_residual],
            [self._visualize_signed_field(image, enhanced_residual_scale).cpu() for image in enhanced_residual],
        ]
        canvas = self._build_preview_canvas(rows)
        caption = (
            "rows=clean/noisy/mean/projected/enhanced/noisy_minus_clean/mean_minus_clean/projected_minus_clean/enhanced_minus_clean, "
            "cols=epsilon sweep for validation sample 0, "
            f"epsilons={[round(value.item(), 6) for value in epsilon]}, "
            f"noisy_residual_scale={noisy_residual_scale.item():.4g}, "
            f"mean_residual_scale={mean_residual_scale.item():.4g}, "
            f"projected_residual_scale={projected_residual_scale.item():.4g}, "
            f"enhanced_residual_scale={enhanced_residual_scale.item():.4g}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/adversarial_by_epsilon": wandb.Image(canvas, caption=caption),
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
        top_row_label = "projector mean images" if "projector" in log_key else "input images"
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
        base_image = self._denormalize_image(clean_images[0]).clamp(0.0, 1.0).cpu()
        rows.append([base_image.clone() for _ in range(epsilon.shape[0])])

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
        top_row_label = "projector mean image" if "projector" in log_key else "input image"
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
        denoiser_optimizer, generator_optimizer = self.optimizers()
        denoiser_scheduler, generator_scheduler = self.lr_schedulers()
        epsilon = self.sample_epsilon(images.shape[0], images.device).to(dtype=images.dtype)
        latent_white_noise = torch.randn(
            images.shape[0],
            self.config.rank,
            device=images.device,
            dtype=images.dtype,
        )

        denoiser_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        loss, metrics = self._run_adversarial_round(
            images,
            epsilon=epsilon,
            latent_white_noise=latent_white_noise,
        )

        self.manual_backward(loss)
        denoiser_optimizer.step()
        generator_optimizer.step()
        denoiser_scheduler.step()
        generator_scheduler.step()

        self.log("train/loss", loss, prog_bar=True, batch_size=images.shape[0])
        self.log(
            "train/denoiser_loss",
            loss.detach(),
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train/generator_loss",
            loss.detach(),
            prog_bar=True,
            batch_size=images.shape[0],
        )
        for name, value in metrics.items():
            self.log(f"train/denoiser_{name}", value, prog_bar=False, batch_size=images.shape[0])
        for name, value in metrics.items():
            self.log(f"train/generator_{name}", value, prog_bar=False, batch_size=images.shape[0])
        self.log("train/denoiser_lr", denoiser_optimizer.param_groups[0]["lr"], prog_bar=False, batch_size=images.shape[0])
        self.log("train/generator_lr", generator_optimizer.param_groups[0]["lr"], prog_bar=False, batch_size=images.shape[0])
        return loss.detach()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        sample_preview = self._build_sample_preview()
        epsilon_preview = self._build_epsilon_preview()
        if sample_preview is None or epsilon_preview is None:
            return

        self._log_image_comparison_grid(sample_preview)
        self._log_image_epsilon_grid(epsilon_preview)
        self._log_basis_comparison_grid(
            singular_vectors=sample_preview["generated_singular_vectors"],
            singular_values=sample_preview["generated_singular_values"],
            clean_images=sample_preview["clean_images"],
            epsilon=sample_preview["epsilon"],
            log_key="val/generated_basis_vectors",
            basis_label="generator normalized basis",
        )
        self._log_basis_epsilon_grid(
            singular_vectors=epsilon_preview["generated_singular_vectors"],
            singular_values=epsilon_preview["generated_singular_values"],
            clean_images=epsilon_preview["clean_images"],
            epsilon=epsilon_preview["epsilon"],
            log_key="val/generated_basis_vectors_by_epsilon",
            basis_label="generator normalized basis",
        )
        self._log_basis_comparison_grid(
            singular_vectors=sample_preview["projector_singular_vectors"],
            singular_values=sample_preview["projector_singular_values"],
            clean_images=sample_preview["mean_images"],
            epsilon=sample_preview["epsilon"],
            log_key="val/projector_basis_vectors",
            basis_label="projector centered basis around mean image",
        )
        self._log_basis_epsilon_grid(
            singular_vectors=epsilon_preview["projector_singular_vectors"],
            singular_values=epsilon_preview["projector_singular_values"],
            clean_images=epsilon_preview["mean_images"],
            epsilon=epsilon_preview["epsilon"],
            log_key="val/projector_basis_vectors_by_epsilon",
            basis_label="projector centered basis around mean image",
        )

    def configure_optimizers(self):
        denoiser_optimizer = torch.optim.AdamW(
            list(self.projector.parameters()) + list(self.enhancer.parameters()) + [self.projector_log_var, self.enhancer_log_var],
            lr=self.config.denoiser_learning_rate,
            weight_decay=self.config.denoiser_weight_decay,
        )
        generator_optimizer = torch.optim.AdamW(
            self.noise_generator.parameters(),
            lr=self.config.generator_learning_rate,
            weight_decay=self.config.generator_weight_decay,
            maximize=True,
        )
        denoiser_scheduler = torch.optim.lr_scheduler.LambdaLR(
            denoiser_optimizer,
            lr_lambda=self._warmup_decay_lr_lambda(
                alpha=self.config.denoiser_lr_alpha,
                warmup_steps=self.config.denoiser_warmup_steps,
                scale_steps=self.config.denoiser_lr_scale_steps,
            ),
        )
        generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            generator_optimizer,
            lr_lambda=self._warmup_decay_lr_lambda(
                alpha=self.config.generator_lr_alpha,
                warmup_steps=self.config.generator_warmup_steps,
                scale_steps=self.config.generator_lr_scale_steps,
            ),
        )
        return (
            [denoiser_optimizer, generator_optimizer],
            [
                {"scheduler": denoiser_scheduler, "interval": "step"},
                {"scheduler": generator_scheduler, "interval": "step"},
            ],
        )
