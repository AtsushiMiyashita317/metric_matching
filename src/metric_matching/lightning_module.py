from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import numpy as np
import torch

from metric_matching.data import restore_image_range
from metric_matching.models import MetricBasisNetwork, MetricFactorNetwork, ScoreNetwork
from metric_matching.score_module import (
    load_score_network_checkpoint,
    prediction_to_noise,
    read_score_checkpoint_config,
)


@dataclass
class MetricMatchingConfig:
    image_channels: int = 3
    image_size: int = 64
    rank: int = 100
    base_channels: int = 64
    num_res_blocks: int = 2
    attention_downsample_factor: int = 4
    use_output_bias: bool = True
    output_bias_variance: float = 1e-3
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    epsilon_min: float = 1e-4
    epsilon_max: float = 5e-2
    copies_per_sample: int = 1
    tikhonov_lambda: float = 1e-4
    score_matching_weight: float = 1.0
    projection_weight: float = 0.0
    detach_score_in_metric_loss: bool = False
    score_target: Literal["noise", "mean"] = "noise"
    metric_target: Literal["direction", "destination"] = "direction"
    score_training_mode: Literal["joint", "pretrained_frozen"] = "joint"
    pretrained_score_checkpoint: str | None = None
    scale_input_by_sqrt_one_plus_epsilon: bool = False
    epsilon_input_mode: Literal["log_clamp", "log_one_plus", "identity"] = "log_clamp"
    preview_fields: int = 8
    preview_samples: int = 4
    preview_steps: int = 7
    preview_scale: float = 0.25
    preview_rk4_substeps: int = 8


class MetricMatchingModule(L.LightningModule):
    def __init__(self, config: MetricMatchingConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.copies_per_sample < 1:
            raise ValueError(f"copies_per_sample must be at least 1, got {self.config.copies_per_sample}")
        if self.config.score_target not in {"noise", "mean"}:
            raise ValueError(
                "score_target must be one of {'noise', 'mean'}, "
                f"got {self.config.score_target}"
            )
        if self.config.metric_target not in {"direction", "destination"}:
            raise ValueError(
                "metric_target must be one of {'direction', 'destination'}, "
                f"got {self.config.metric_target}"
            )
        if self.config.score_training_mode not in {"joint", "pretrained_frozen"}:
            raise ValueError(
                "score_training_mode must be one of {'joint', 'pretrained_frozen'}, "
                f"got {self.config.score_training_mode}"
            )
        if self.config.score_training_mode == "pretrained_frozen" and self.config.pretrained_score_checkpoint is None:
            raise ValueError("pretrained_score_checkpoint is required when score_training_mode='pretrained_frozen'.")
        self.save_hyperparameters(asdict(config))
        self.loaded_score_checkpoint_path: str | None = None
        self.loaded_score_scaling_mode: bool | None = None
        self.loaded_score_epsilon_input_mode: str | None = None
        self.network: MetricFactorNetwork | None = None
        self.metric_network: MetricBasisNetwork | None = None
        self.score_network: ScoreNetwork | None = None
        if self.uses_joint_score_prediction:
            self.network = MetricFactorNetwork(
                image_size=config.image_size,
                in_channels=config.image_channels,
                rank=config.rank,
                base_channels=config.base_channels,
                num_res_blocks=config.num_res_blocks,
                attention_downsample_factor=config.attention_downsample_factor,
                use_output_bias=config.use_output_bias,
                output_bias_variance=config.output_bias_variance,
                scale_input_by_sqrt_one_plus_epsilon=config.scale_input_by_sqrt_one_plus_epsilon,
                epsilon_input_mode=config.epsilon_input_mode,
            )
        else:
            score_checkpoint_config = read_score_checkpoint_config(
                Path(self.config.pretrained_score_checkpoint)
            )
            self.metric_network = MetricBasisNetwork(
                image_size=config.image_size,
                in_channels=config.image_channels,
                rank=config.rank,
                base_channels=config.base_channels,
                num_res_blocks=config.num_res_blocks,
                attention_downsample_factor=config.attention_downsample_factor,
                use_output_bias=config.use_output_bias,
                output_bias_variance=config.output_bias_variance,
                scale_input_by_sqrt_one_plus_epsilon=config.scale_input_by_sqrt_one_plus_epsilon,
                epsilon_input_mode=config.epsilon_input_mode,
            )
            self.score_network = ScoreNetwork(
                image_size=config.image_size,
                in_channels=config.image_channels,
                base_channels=config.base_channels,
                num_res_blocks=config.num_res_blocks,
                attention_downsample_factor=config.attention_downsample_factor,
                use_output_bias=config.use_output_bias,
                output_bias_variance=config.output_bias_variance,
                scale_input_by_sqrt_one_plus_epsilon=bool(
                    score_checkpoint_config["scale_input_by_sqrt_one_plus_epsilon"]
                ),
                epsilon_input_mode=str(score_checkpoint_config["epsilon_input_mode"]),
            )
            checkpoint_metadata = load_score_network_checkpoint(
                self.score_network,
                checkpoint_path=Path(self.config.pretrained_score_checkpoint),
            )
            self.loaded_score_checkpoint_path = str(checkpoint_metadata["checkpoint_path"])
            self.loaded_score_scaling_mode = bool(checkpoint_metadata["scale_input_by_sqrt_one_plus_epsilon"])
            self.loaded_score_epsilon_input_mode = str(checkpoint_metadata["epsilon_input_mode"])
            self.score_network.requires_grad_(False)
            self.score_network.eval()
        self.example_input_array = (
            torch.randn(2, config.image_channels, config.image_size, config.image_size),
            torch.full((2,), 1e-2),
        )

    @property
    def uses_joint_score_prediction(self) -> bool:
        return self.config.score_training_mode == "joint"

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.uses_joint_score_prediction and self.score_network is not None:
            self.score_network.eval()
        return self

    def forward(self, image: torch.Tensor, epsilon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_joint_score_prediction:
            assert self.network is not None
            metric_basis_prediction, score_prediction = self.network(image, epsilon)
        else:
            assert self.metric_network is not None
            assert self.score_network is not None
            metric_basis_prediction = self.metric_network(image, epsilon)
            with torch.no_grad():
                score_prediction = self.score_network(image, epsilon)
        score = self._prediction_to_score(score_prediction, image, epsilon)
        metric_basis = self._prediction_to_metric_basis(metric_basis_prediction, image, score, epsilon)
        return metric_basis, score

    def _forward_metric_single(self, image: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        # image: [C, H, W], epsilon: [], returns metric_factors: [K, C, H, W]
        metric_factors, _ = self.forward(image.unsqueeze(0), epsilon.unsqueeze(0))
        return metric_factors[0]

    def _forward_metric_and_tangent_single(
        self, 
        image: torch.Tensor, 
        tangent: torch.Tensor,
        epsilon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image/tangent: [C, H, W], epsilon: [], returns metric_factors: [K, C, H, W], metric_derivatives: [K, C, H, W]
        metric_factors, metric_derivatives = torch.func.jvp(
            lambda img: self._forward_metric_single(img, epsilon),
            (image,),
            (tangent,),
        )
        return metric_factors, metric_derivatives
    
    def forward_metric_and_tangent(
        self,
        image: torch.Tensor,
        tangent: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # image/tangent: [B, C, H, W], epsilon: [B], returns metric_factors/metric_derivatives: [B, K, C, H, W]
        metric_factors, metric_derivatives = torch.func.vmap(self._forward_metric_and_tangent_single)(
            image,
            tangent,
            epsilon,
        )
        return metric_factors, metric_derivatives
    
    def sample_epsilon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps_min = torch.tensor(self.config.epsilon_min, device=device).log()
        eps_max = torch.tensor(self.config.epsilon_max, device=device).log()
        return torch.exp(torch.rand(batch_size, device=device) * (eps_max - eps_min) + eps_min)

    def _prediction_to_score(
        self,
        prediction: torch.Tensor,
        noisy_images: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        return prediction_to_noise(
            prediction=prediction,
            noisy_images=noisy_images,
            epsilon=epsilon,
            score_target=self.config.score_target,
        )

    def _prediction_to_metric_basis(
        self,
        metric_basis_prediction: torch.Tensor,
        noisy_images: torch.Tensor,
        score: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        _, _, channels, height, width = metric_basis_prediction.shape
        if self.config.metric_target == "direction":
            return metric_basis_prediction / (channels * height * width) ** 0.5
        if self.config.detach_score_in_metric_loss:
            score = score.detach()
        mean_images = noisy_images - score * epsilon.sqrt()[:, None, None, None]
        return (metric_basis_prediction - mean_images[:, None, :, :, :])

    def compute_low_rank_loss(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.copies_per_sample > 1:
            # images: [B, C, H, W] -> [B * copies_per_sample, C, H, W]
            images = images.repeat_interleave(self.config.copies_per_sample, dim=0)

        batch_size = images.shape[0]
        epsilon = self.sample_epsilon(batch_size, images.device)
        # epsilon: [B], noise/noisy_images/delta: [B, C, H, W]
        noise = torch.randn_like(images)
        noisy_images = images + epsilon.sqrt()[:, None, None, None] * noise
        # metric_basis: [B, rank, C, H, W], score: [B, C, H, W]
        metric_basis, score = self.forward(noisy_images, epsilon)

        # metric_flat: [B, rank, C * H * W], score_flat/delta_flat: [B, C * H * W]
        metric_flat = metric_basis.flatten(start_dim=2)
        score_flat = score.flatten(start_dim=1)
        metric_loss_score_flat = score_flat.detach() if self.config.detach_score_in_metric_loss else score_flat
        noise_flat = noise.flatten(start_dim=1)
        factor_rank = self.config.rank
        basis_rank = metric_flat.shape[1]
        data_dim = metric_flat.shape[2]
        normalization = float(data_dim**2)

        # metric_gram: [B, rank, rank], metric_score_dot: [B, rank]
        metric_gram = torch.matmul(metric_flat, metric_flat.transpose(1, 2))
        metric_frob = metric_gram.square().sum(dim=(1, 2)) / normalization
        score_gram = metric_loss_score_flat.square().sum(dim=1)
        score_frob = score_gram.square() / normalization
        metric_score_dot = torch.matmul(metric_flat, metric_loss_score_flat[:, :, None]).squeeze(2)
        metric_score = 2 * metric_score_dot.square().sum(dim=1) / normalization
        frob_term = metric_frob + score_frob + metric_score

        # metric_delta_dot: [B, rank], score_delta_dot/target_sq_norm: [B]
        metric_delta_dot = torch.matmul(metric_flat, noise_flat[:, :, None]).squeeze(2)
        metric_delta = 2 * metric_delta_dot.square().sum(dim=1) / normalization
        score_delta_dot = (metric_loss_score_flat * noise_flat).sum(dim=1)
        score_delta = 2 * score_delta_dot.square() / normalization
        alignment_term = metric_delta + score_delta

        metric_gram3 = torch.matmul(metric_gram, metric_flat)
        metric_frob3 = metric_gram3.square().sum(dim=(1, 2)) / normalization
        metric_gram4 = torch.matmul(metric_gram, metric_gram)
        metric_frob4 = metric_gram4.square().sum(dim=(1, 2)) / normalization
        projection_term = metric_frob - 2 * metric_frob3 + metric_frob4

        target_sq_norm = noise_flat.square().sum(dim=1)
        target_term = target_sq_norm.square() / normalization
        tangent_dim = metric_flat.square().sum(dim=(1, 2))
        score_norm = metric_loss_score_flat.square().sum(dim=1)
        reg_term = self.config.tikhonov_lambda * 2.0 * (tangent_dim + score_norm) / normalization
        score_matching_term = (score_flat - noise_flat).square().mean(dim=1)

        metric_loss = (frob_term + reg_term - alignment_term).mean()
        score_loss = self.config.score_matching_weight * score_matching_term.mean()
        projection_loss = self.config.projection_weight * projection_term.mean()
        optimized_score_loss = score_loss if self.uses_joint_score_prediction else torch.zeros_like(score_loss)
        loss = metric_loss + optimized_score_loss + projection_loss
        target_norm = (target_sq_norm / data_dim).mean()
        factor_norm = metric_flat[:, :factor_rank].square().sum(dim=(1, 2)).div(factor_rank * data_dim).mean()
        mean_offset_norm = metric_flat[:, factor_rank:].square().sum(dim=(1, 2)).div(data_dim).mean()

        metrics = {
            "metric_loss": metric_loss.detach(),
            "score_matching_loss": score_loss.detach(),
            "optimized_score_matching_loss": optimized_score_loss.detach(),
            "projection_loss": projection_loss.detach(),
            "frob_term": frob_term.mean().detach(),
            "alignment_term": alignment_term.mean().detach(),
            "target_term": target_term.mean().detach(),
            "reg_term": reg_term.mean().detach(),
            "score_matching_term": score_matching_term.mean().detach(),
            "tangent_dim": tangent_dim.mean().detach(),
            "epsilon_mean": epsilon.mean().detach(),
            "effective_batch_size": torch.tensor(float(batch_size), device=images.device),
            "target_delta_norm": target_norm.detach(),
            "factor_norm": factor_norm.detach(),
            "mean_offset_norm": mean_offset_norm.detach(),
            "factor_rank": torch.tensor(float(factor_rank), device=images.device),
            "metric_basis_rank": torch.tensor(float(basis_rank), device=images.device),
            "data_dim": torch.tensor(float(data_dim), device=images.device),
            "loss_scale_denominator": torch.tensor(normalization, device=images.device),
            "score_mode": torch.tensor(
                0.0 if self.config.score_target == "noise" else 1.0,
                device=images.device,
            ),
            "score_training_mode": torch.tensor(
                0.0 if self.uses_joint_score_prediction else 1.0,
                device=images.device,
            ),
            "metric_mode": torch.tensor(
                0.0 if self.config.metric_target == "direction" else 1.0,
                device=images.device,
            ),
        }
        return loss, metrics

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        images = batch["image"]
        loss, metrics = self.compute_low_rank_loss(images)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=images.shape[0])
        for name, value in metrics.items():
            self.log(f"{stage}/{name}", value, prog_bar=False, batch_size=images.shape[0])
        return loss

    def _denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        datamodule = getattr(self.trainer, "datamodule", None)
        stats = getattr(datamodule, "stats", None)
        return restore_image_range(image, stats=stats)

    def _align_eigenvectors(
        self,
        eigenvectors: torch.Tensor,
        reference_eigenvectors: torch.Tensor | None,
    ) -> torch.Tensor:
        # eigenvectors/reference_eigenvectors: [K, C, H, W]
        if reference_eigenvectors is None:
            return eigenvectors

        # flat_current/flat_reference: [K, C * H * W], similarity: [K, K]
        flat_current = eigenvectors.flatten(start_dim=1)
        flat_reference = reference_eigenvectors.flatten(start_dim=1)
        similarity = torch.matmul(flat_reference, flat_current.transpose(0, 1))

        assigned: set[int] = set()
        permutation: list[int] = []
        sign_flips: list[float] = []
        for ref_idx in range(flat_reference.shape[0]):
            row = similarity[ref_idx].clone()
            if assigned:
                row[list(assigned)] = float("-inf")
            best_idx = int(row.abs().argmax().item())
            assigned.add(best_idx)
            permutation.append(best_idx)
            sign_flips.append(1.0 if similarity[ref_idx, best_idx].item() >= 0.0 else -1.0)

        ordered = eigenvectors[permutation]
        sign = torch.tensor(sign_flips, device=ordered.device, dtype=ordered.dtype).view(-1, 1, 1, 1)
        return ordered * sign

    def _top_metric_eigenvectors_single(
        self,
        normalized_image: torch.Tensor,
        epsilon: torch.Tensor,
        reference_eigenvectors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # normalized_image: [1, C, H, W], epsilon: [1]
        basis_fields, _ = self.forward(normalized_image, epsilon)
        eigenvectors, eigenvalues = self._top_metric_eigenvectors(basis_fields)
        # eigenvectors[0]: [K, C, H, W], eigenvalues[0]: [K]
        eigenvectors = self._align_eigenvectors(eigenvectors[0], reference_eigenvectors)
        eigenvalues = eigenvalues[0]
        return eigenvectors, eigenvalues

    def _evaluate_tracked_eigenvector_field(
        self,
        normalized_image: torch.Tensor,
        epsilon: torch.Tensor,
        field_idx: int,
        reference_eigenvectors: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eigenvectors, _ = self._top_metric_eigenvectors_single(
            normalized_image=normalized_image,
            epsilon=epsilon,
            reference_eigenvectors=reference_eigenvectors,
        )
        # returns tracked field: [1, C, H, W], all eigenvectors: [K, C, H, W]
        return eigenvectors[field_idx].unsqueeze(0), eigenvectors

    def _integrate_basis_field_rk4(
        self,
        base_image: torch.Tensor,
        epsilon: torch.Tensor,
        field_idx: int,
        target_time: float,
        scale_factor: torch.Tensor,
        reference_eigenvectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        substeps = max(1, int(abs(target_time) * self.config.preview_rk4_substeps))
        dt = target_time / substeps
        # state/base_image: [1, C, H, W], each k*: [1, C, H, W]
        state = base_image.clone()
        tracked_eigenvectors = reference_eigenvectors

        for _ in range(substeps):
            field_1, tracked_1 = self._evaluate_tracked_eigenvector_field(state, epsilon, field_idx, tracked_eigenvectors)
            k1 = scale_factor * field_1
            field_2, tracked_2 = self._evaluate_tracked_eigenvector_field(
                state + 0.5 * dt * k1,
                epsilon,
                field_idx,
                tracked_1,
            )
            k2 = scale_factor * field_2
            field_3, tracked_3 = self._evaluate_tracked_eigenvector_field(
                state + 0.5 * dt * k2,
                epsilon,
                field_idx,
                tracked_2,
            )
            k3 = scale_factor * field_3
            field_4, tracked_4 = self._evaluate_tracked_eigenvector_field(
                state + dt * k3,
                epsilon,
                field_idx,
                tracked_3,
            )
            k4 = scale_factor * field_4
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            tracked_eigenvectors, _ = self._top_metric_eigenvectors_single(
                normalized_image=state,
                epsilon=epsilon,
                reference_eigenvectors=tracked_4,
            )

        return state
    
    def evaluate_geodesic_direction(
        self,
        image_velocity: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        image, velocity = image_velocity.select(-1, 0), image_velocity.select(-1, 1)
        metric_factors, metric_derivatives = self.forward_metric_and_tangent(image, velocity, epsilon)
        a1 = torch.einsum("bkcij,bcij->bk", metric_derivatives, velocity)
        a1 = torch.einsum("bk,bkcij->bcij", a1, metric_factors)
        a2 = torch.einsum("bkcij,bcij->bk", metric_factors, velocity)
        a2 = torch.einsum("bk,bkcij->bcij", a2, metric_derivatives)
        acceleration = a1 + a2
        derivative = torch.stack([velocity, acceleration], dim=-1)
        return derivative

    def integrate_geodesic_rk4(
        self,
        initial_image: torch.Tensor,
        initial_velocity: torch.Tensor,
        epsilon: torch.Tensor,
        time_per_step: float,
        steps: int,
        scale_factor: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        substeps = self.config.preview_rk4_substeps
        dt = time_per_step / substeps
        state = torch.stack([initial_image, initial_velocity], dim=-1)
        image_list = [initial_image]
        velocity_list = [initial_velocity]
        for _ in range(steps):
            for _ in range(substeps):
                d1 = self.evaluate_geodesic_direction(state, epsilon)
                k1 = scale_factor * d1
                d2 = self.evaluate_geodesic_direction(state + 0.5 * dt * k1, epsilon)
                k2 = scale_factor * d2
                d3 = self.evaluate_geodesic_direction(state + 0.5 * dt * k2, epsilon)
                k3 = scale_factor * d3
                d4 = self.evaluate_geodesic_direction(state + dt * k3, epsilon)
                k4 = scale_factor * d4
                state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            image_list.append(state.select(-1, 0))
            velocity_list.append(state.select(-1, 1))
        return image_list, velocity_list

    def _preview_geodesic_images(
        self,
        initial_image: torch.Tensor,
        initial_velocity: torch.Tensor,
        epsilon: torch.Tensor,
        scale_factor: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], tuple[int, int]]:
        left_steps = self.config.preview_steps // 2
        right_steps = self.config.preview_steps - 1 - left_steps

        negative_images, negative_velocities = self.integrate_geodesic_rk4(
            initial_image=initial_image,
            initial_velocity=-initial_velocity,
            epsilon=epsilon,
            time_per_step=1.0,
            steps=left_steps,
            scale_factor=scale_factor,
        )
        positive_images, positive_velocities = self.integrate_geodesic_rk4(
            initial_image=initial_image,
            initial_velocity=initial_velocity,
            epsilon=epsilon,
            time_per_step=1.0,
            steps=right_steps,
            scale_factor=scale_factor,
        )
        images = list(reversed(negative_images[1:])) + positive_images
        velocities = list(reversed(negative_velocities[1:])) + positive_velocities
        return images, velocities, (-left_steps, right_steps)

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

    def _visualize_vector_field(
        self,
        vector_field: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        # Map zero displacement to mid-gray so sign structure is visible.
        image = 0.5 + 0.5 * (vector_field / scale.clamp_min(1e-6))
        return image.clamp(0.0, 1.0)

    def _top_metric_eigenvectors(self, basis_fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # basis_fields: [B, K, C, H, W]
        flat_basis = basis_fields.flatten(start_dim=2)
        _, singular_values, vh = torch.linalg.svd(flat_basis, full_matrices=False)
        # singular_values: [B, K], vh/eigenvectors: [B, K, C * H * W] -> [B, K, C, H, W]
        eigenvectors = vh.view(vh.shape[0], vh.shape[1], *basis_fields.shape[2:])
        eigenvalues = singular_values.square()
        return eigenvectors, eigenvalues

    def _log_vector_field_grid(self) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return

        num_samples = min(self.config.preview_samples, len(val_dataset))
        samples = [val_dataset[idx]["image"] for idx in range(num_samples)]
        # normalized_batch: [B, C, H, W], epsilon: [B]
        normalized_batch = torch.stack(samples, dim=0).to(self.device)
        epsilon = self._preview_epsilon_values(
            num_values=1,
            device=self.device,
            dtype=normalized_batch.dtype,
        ).expand(num_samples)
        epsilon_value = epsilon[0].item()

        with torch.no_grad():
            basis_fields, _ = self.forward(normalized_batch, epsilon)
            eigenvectors, eigenvalues = self._top_metric_eigenvectors(basis_fields)

        num_fields = min(self.config.preview_fields, eigenvectors.shape[1])
        rows: list[list[torch.Tensor]] = []
        rows.append([self._denormalize_image(image).clamp(0.0, 1.0).cpu() for image in normalized_batch])

        displayed_eigenvalues = []
        for field_idx in range(num_fields):
            row_fields = eigenvectors[:, field_idx]
            row_scale = row_fields.square().mean(dim=(1, 2, 3)).sqrt().max().clamp_min(1e-6)
            row_images = [
                self._visualize_vector_field(row_fields[sample_idx], row_scale).cpu()
                for sample_idx in range(num_samples)
            ]
            rows.append(row_images)
            displayed_eigenvalues.append(eigenvalues[:, field_idx].mean().item())

        canvas = self._build_preview_canvas(rows)
        caption = (
            f"top row=input images, lower rows=top metric eigenvectors 0..{num_fields - 1}, "
            f"cols=validation samples 0..{num_samples - 1}, epsilon={epsilon_value:.4g}, "
            f"mean_eigenvalues={[round(v, 4) for v in displayed_eigenvalues]}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/vector_fields": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_vector_field_epsilon_grid(self) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return

        num_epsilons = max(1, self.config.preview_samples)
        normalized_image = val_dataset[0]["image"].unsqueeze(0).to(self.device)
        epsilon = self._preview_epsilon_values(
            num_values=num_epsilons,
            device=self.device,
            dtype=normalized_image.dtype,
        )
        repeated_image = normalized_image.expand(num_epsilons, -1, -1, -1)

        with torch.no_grad():
            basis_fields, _ = self.forward(repeated_image, epsilon)
            eigenvectors, eigenvalues = self._top_metric_eigenvectors(basis_fields)

        num_fields = min(self.config.preview_fields, eigenvectors.shape[1])
        rows: list[list[torch.Tensor]] = []
        base_image = self._denormalize_image(normalized_image[0]).clamp(0.0, 1.0).cpu()
        rows.append([base_image.clone() for _ in range(num_epsilons)])

        displayed_eigenvalues = []
        for field_idx in range(num_fields):
            row_fields = eigenvectors[:, field_idx]
            row_scale = row_fields.square().mean(dim=(1, 2, 3)).sqrt().max().clamp_min(1e-6)
            row_images = [
                self._visualize_vector_field(row_fields[epsilon_idx], row_scale).cpu()
                for epsilon_idx in range(num_epsilons)
            ]
            rows.append(row_images)
            displayed_eigenvalues.append([round(v.item(), 4) for v in eigenvalues[:, field_idx]])

        canvas = self._build_preview_canvas(rows)
        epsilon_values = [round(v.item(), 6) for v in epsilon]
        caption = (
            f"top row=input image, lower rows=top metric eigenvectors 0..{num_fields - 1}, "
            f"cols=epsilon sweep for validation sample 0, epsilons={epsilon_values}, "
            f"eigenvalues_by_row={displayed_eigenvalues}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/vector_fields_by_epsilon": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def _log_vector_field_preview(self) -> None:
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None or len(val_dataset) == 0:
            return

        sample = val_dataset[0]
        normalized_image = sample["image"].unsqueeze(0).to(self.device)
        epsilon_value = (self.config.epsilon_min * self.config.epsilon_max) ** 0.5
        epsilon = torch.full((1,), epsilon_value, device=self.device, dtype=normalized_image.dtype)

        with torch.no_grad():
            eigenvectors, eigenvalues = self._top_metric_eigenvectors_single(normalized_image, epsilon)

        num_fields = min(self.config.preview_fields, eigenvectors.shape[0])
        rows: list[list[torch.Tensor]] = []
        velocity_rows: list[list[torch.Tensor]] = []
        step_bounds: tuple[int, int] | None = None
        for field_idx in range(num_fields):
            vector_field = eigenvectors[field_idx]
            rms = vector_field.square().mean().sqrt().clamp_min(1e-6)
            scale_factor = torch.as_tensor(
                self.config.preview_scale,
                device=self.device,
                dtype=normalized_image.dtype,
            ) / rms
            geodesic_images, geodesic_velocities, step_bounds = self._preview_geodesic_images(
                initial_image=normalized_image,
                initial_velocity=vector_field.unsqueeze(0),
                epsilon=epsilon,
                scale_factor=scale_factor,
            )
            row_images = [self._denormalize_image(image)[0].clamp(0.0, 1.0).cpu() for image in geodesic_images]
            velocity_scale = torch.stack(geodesic_velocities, dim=0).square().mean(dim=(1, 2, 3, 4)).sqrt().max()
            velocity_scale = velocity_scale.clamp_min(1e-6)
            velocity_row = [
                self._visualize_vector_field(velocity[0], velocity_scale).cpu() for velocity in geodesic_velocities
            ]
            rows.append(row_images)
            velocity_rows.append(velocity_row)

        canvas = self._build_preview_canvas(rows)
        velocity_canvas = self._build_preview_canvas(velocity_rows)
        assert step_bounds is not None
        caption = (
            f"rows=geodesics initialized from top metric eigenvectors 0..{num_fields - 1}, "
            f"cols=geodesic steps in [{step_bounds[0]}, {step_bounds[1]}], "
            f"epsilon={epsilon_value:.4g}, rk4_substeps={self.config.preview_rk4_substeps}, "
            f"initial_eigenvalues={[round(v.item(), 4) for v in eigenvalues[:num_fields]]}"
        )
        velocity_caption = (
            f"rows=geodesic velocities initialized from top metric eigenvectors 0..{num_fields - 1}, "
            f"cols=geodesic steps in [{step_bounds[0]}, {step_bounds[1]}], "
            f"epsilon={epsilon_value:.4g}, rk4_substeps={self.config.preview_rk4_substeps}, "
            f"initial_eigenvalues={[round(v.item(), 4) for v in eigenvalues[:num_fields]]}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/vector_field_integrations": wandb.Image(canvas, caption=caption),
                    "val/vector_field_velocity_integrations": wandb.Image(
                        velocity_canvas,
                        caption=velocity_caption,
                    ),
                },
                step=self.global_step,
            )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._log_vector_field_grid()
        self._log_vector_field_epsilon_grid()
        self._log_vector_field_preview()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
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
