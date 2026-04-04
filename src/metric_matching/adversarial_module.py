from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import numpy as np
import torch

from metric_matching.data import restore_image_range
from metric_matching.models import MetricFactorNetwork, MetricBasisNetwork, ScoreNetwork


@dataclass
class AdversarialDenoisingConfig:
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
    scale_input: bool = False
    epsilon_input_mode: str = "log_clamp"
    preview_samples: int = 4


class AdversarialDenoisingModule(L.LightningModule):
    def __init__(self, config: AdversarialDenoisingConfig) -> None:
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

    def _project_noisy_images(
        self,
        images: torch.Tensor,
        noisy_images: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        basis, mean = self.projector(noisy_images, epsilon)
        var_basis = basis - mean.unsqueeze(1)
        covariance = torch.einsum("bmchw,bnchw->bmn", var_basis, var_basis) + torch.eye(covariance.shape[-1], device=covariance.device) * 1e-6
        covariance_inv = torch.linalg.inv(covariance)

        noisy_diff = noisy_images - mean
        noisy_latent = torch.einsum("bnchw,bchw->bn", var_basis, noisy_diff)
        noisy_latent = torch.einsum("bmn,bn->bm", covariance_inv, noisy_latent)

        clean_diff = images - mean
        clean_latent = torch.einsum("bnchw,bchw->bn", var_basis, clean_diff)
        clean_latent = torch.einsum("bmn,bn->bm", covariance_inv, clean_latent)
        projected_images = mean + torch.einsum("bm,bmchw->bchw", clean_latent, var_basis)

        return projected_images, {
            "covariance": covariance,
            "covariance_inv": covariance_inv,
            "latent": noisy_latent,
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
        noisy_images: torch.Tensor,
        epsilon: torch.Tensor
    ) -> torch.Tensor:
        data_dim = self.config.image_channels * self.config.image_size * self.config.image_size
        tangent_dim = self.config.rank
        normal_dim = data_dim - tangent_dim

        projected_images, aux = self._project_noisy_images(images, noisy_images, epsilon)
        covariance = aux["covariance"]
        latent = aux["latent"]
        project_trace_term = latent.square().sum(dim=1).div(epsilon[:, None] * self.projector_log_var.exp())
        project_logdet_term = torch.logdet(covariance) + tangent_dim * (torch.log(epsilon) + self.projector_log_var)
        project_nll = 0.5 * project_trace_term + 0.5 * project_logdet_term

        enhanced_images = self._enhance_images(noisy_images, projected_images, epsilon)
        enhance_trace_term = (enhanced_images - images).square().sum(dim=(1, 2, 3)).div(epsilon * self.enhancer_log_var.exp())
        enhance_logdet_term = normal_dim * (torch.log(epsilon) + self.enhancer_log_var)
        enhance_nll = 0.5 * enhance_trace_term + 0.5 * enhance_logdet_term

        nll = project_nll + enhance_nll
        return nll, {
            "project_nll": project_nll,
            "enhance_nll": enhance_nll,
            "projected_images": projected_images,
            "enhanced_images": enhanced_images,
            "mean": aux["mean"],
            "basis": aux["basis"],
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
        noisy_images, generated = self._generate_noise(images, epsilon, latent_white_noise=latent_white_noise)
        nll, aux = self._compute_nll(images, noisy_images, epsilon)
        projected_images = aux["projected_images"]
        enhanced_images = aux["enhanced_images"]
        mean = aux["mean"]
        basis = aux["basis"]

        metrics = {
            "nll": nll.mean().detach(),
            "generator_basis_rms_before_norm": generated["basis_rms_before_norm"].mean().detach(),
            "generator_basis_rms_after_norm": generated["normalized_basis"]
            .square()
            .mean(dim=(2, 3, 4))
            .sqrt()
            .mean()
            .detach(),
            "structured_noise_rms": generated["structured_noise"].square().mean().sqrt().detach(),
            "perturbation_rms": generated["perturbation"].square().mean().sqrt().detach(),
            "projector_log_var": self.projector_log_var.detach(),
            "enhancer_log_var": self.enhancer_log_var.detach(),
        }
        aux = {
            "epsilon": epsilon,
            "noisy_images": noisy_images,
            "projected_images": projected_images,
            "enhanced_images": enhanced_images,
            "mean": mean,
            "basis": basis,
        }
        return nll, metrics, aux

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        nll, metrics, aux = self._run_adversarial_round(images)
        loss = nll.mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=images.shape[0])
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, prog_bar=False, batch_size=images.shape[0])
        return loss

    def _denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        datamodule = getattr(self.trainer, "datamodule", None)
        stats = getattr(datamodule, "stats", None)
        return restore_image_range(image, stats=stats)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        denoiser_optimizer, generator_optimizer = self.optimizers()
        epsilon = self.sample_epsilon(images.shape[0], images.device).to(dtype=images.dtype)
        latent_white_noise = torch.randn(
            images.shape[0],
            self.config.rank,
            device=images.device,
            dtype=images.dtype,
        )

        self.toggle_optimizer(denoiser_optimizer)
        denoiser_optimizer.zero_grad()
        denoiser_loss, denoiser_metrics, _ = self._run_adversarial_round(
            images,
            epsilon=epsilon,
            latent_white_noise=latent_white_noise,
        )
        self.manual_backward(denoiser_loss)
        denoiser_optimizer.step()
        self.untoggle_optimizer(denoiser_optimizer)

        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()
        generator_reconstruction_loss, generator_metrics, _ = self._run_adversarial_round(
            images,
            epsilon=epsilon,
            latent_white_noise=latent_white_noise,
        )
        generator_loss = -self.config.generator_loss_weight * generator_reconstruction_loss
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)

        self.log("train/loss", denoiser_loss, prog_bar=True, batch_size=images.shape[0])
        self.log(
            "train/denoiser_loss",
            denoiser_loss.detach(),
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train/generator_loss",
            generator_loss.detach(),
            prog_bar=True,
            batch_size=images.shape[0],
        )
        for name, value in denoiser_metrics.items():
            self.log(f"train/denoiser_{name}", value, prog_bar=False, batch_size=images.shape[0])
        for name, value in generator_metrics.items():
            self.log(f"train/generator_{name}", value, prog_bar=False, batch_size=images.shape[0])
        return denoiser_loss.detach()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, list):
            for scheduler in schedulers:
                scheduler.step()
        else:
            schedulers.step()

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
        )
        denoiser_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            denoiser_optimizer,
            T_max=max(self.trainer.max_epochs, 1),
        )
        generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            generator_optimizer,
            T_max=max(self.trainer.max_epochs, 1),
        )
        return [denoiser_optimizer, generator_optimizer], [denoiser_scheduler, generator_scheduler]
