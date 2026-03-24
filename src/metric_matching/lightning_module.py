from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import numpy as np
import torch

from metric_matching.models import MetricFactorNetwork


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
        self.save_hyperparameters(asdict(config))
        self.network = MetricFactorNetwork(
            in_channels=config.image_channels,
            rank=config.rank,
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

    def forward(self, image: torch.Tensor, epsilon: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network(image, epsilon)

    def sample_epsilon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps_min = torch.tensor(self.config.epsilon_min, device=device).log()
        eps_max = torch.tensor(self.config.epsilon_max, device=device).log()
        return torch.exp(torch.rand(batch_size, device=device) * (eps_max - eps_min) + eps_min)

    def compute_low_rank_loss(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.config.copies_per_sample > 1:
            images = images.repeat_interleave(self.config.copies_per_sample, dim=0)

        batch_size = images.shape[0]
        epsilon = self.sample_epsilon(batch_size, images.device)
        noise = torch.randn_like(images) * epsilon.sqrt()[:, None, None, None]
        noisy_images = images + noise
        delta = images - noisy_images
        metric_basis, score = self.forward(noisy_images, epsilon)

        metric_flat = metric_basis.flatten(start_dim=2)
        score_flat = score.flatten(start_dim=1)
        delta_flat = delta.flatten(start_dim=1)
        factor_rank = self.config.rank
        basis_rank = metric_flat.shape[1]
        data_dim = metric_flat.shape[2]
        normalization = float(data_dim**2)

        metric_gram = torch.matmul(metric_flat, metric_flat.transpose(1, 2))
        metric_frob = metric_gram.square().sum(dim=(1, 2)) / normalization
        score_gram = score_flat.square().sum(dim=1)
        score_frob = score_gram.square() / normalization
        metric_score_dot = torch.matmul(metric_flat, score_flat[:, :, None]).squeeze(2)
        metric_score = 2 * metric_score_dot.square().sum(dim=1) / normalization
        frob_term = metric_frob + score_frob + metric_score

        metric_delta_dot = torch.matmul(metric_flat, delta_flat[:, :, None]).squeeze(2)
        metric_delta = 2 * metric_delta_dot.square().sum(dim=1) / (epsilon * normalization)
        score_delta_dot = (score_flat * delta_flat).sum(dim=1)
        score_delta = 2 * score_delta_dot.square() / (epsilon * normalization)
        alignment_term = metric_delta + score_delta

        target_sq_norm = delta_flat.square().sum(dim=1)
        target_term = target_sq_norm.square() / (epsilon.square() * normalization)
        tangent_dim = metric_flat.square().sum(dim=(1, 2))
        score_norm = score_flat.square().sum(dim=1)
        reg_term = self.config.tikhonov_lambda * 2.0 * (tangent_dim + score_norm) / normalization
        predicted_score = score_flat
        target_score = delta_flat / epsilon.sqrt()[:, None]
        score_matching_term = (predicted_score - target_score).square().mean(dim=1)

        metric_loss = (frob_term + reg_term - alignment_term).mean()
        score_loss = self.config.score_matching_weight * score_matching_term.mean()
        loss = metric_loss + score_loss
        target_norm = (target_sq_norm / data_dim).mean()
        factor_norm = metric_flat[:, :factor_rank].square().sum(dim=(1, 2)).div(factor_rank * data_dim).mean()
        mean_offset_norm = metric_flat[:, factor_rank:].square().sum(dim=(1, 2)).div(data_dim).mean()

        metrics = {
            "metric_loss": metric_loss.detach(),
            "score_matching_loss": score_loss.detach(),
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
        if stats is None:
            return image
        mean = stats.mean.to(device=image.device, dtype=image.dtype)
        std = stats.std.to(device=image.device, dtype=image.dtype)
        return image * std + mean

    def _align_eigenvectors(
        self,
        eigenvectors: torch.Tensor,
        reference_eigenvectors: torch.Tensor | None,
    ) -> torch.Tensor:
        if reference_eigenvectors is None:
            return eigenvectors

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
        basis_fields, _ = self.forward(normalized_image, epsilon)
        eigenvectors, eigenvalues = self._top_metric_eigenvectors(basis_fields)
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

    def _visualize_vector_field(
        self,
        vector_field: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        # Map zero displacement to mid-gray so sign structure is visible.
        image = 0.5 + 0.5 * (vector_field / scale.clamp_min(1e-6))
        return image.clamp(0.0, 1.0)

    def _top_metric_eigenvectors(self, basis_fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_basis = basis_fields.flatten(start_dim=2)
        _, singular_values, vh = torch.linalg.svd(flat_basis, full_matrices=False)
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
        normalized_batch = torch.stack(samples, dim=0).to(self.device)
        epsilon_value = (self.config.epsilon_min * self.config.epsilon_max) ** 0.5
        epsilon = torch.full((num_samples,), epsilon_value, device=self.device, dtype=normalized_batch.dtype)

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
        time_values = torch.linspace(
            -1.0,
            1.0,
            self.config.preview_steps,
            device=self.device,
            dtype=normalized_image.dtype,
        )

        rows: list[list[torch.Tensor]] = []
        for field_idx in range(num_fields):
            vector_field = eigenvectors[field_idx]
            rms = vector_field.square().mean().sqrt().clamp_min(1e-6)
            scale_factor = torch.as_tensor(
                self.config.preview_scale,
                device=self.device,
                dtype=normalized_image.dtype,
            ) / rms
            row_images = []
            for t in time_values:
                integrated = self._integrate_basis_field_rk4(
                    base_image=normalized_image,
                    epsilon=epsilon,
                    field_idx=field_idx,
                    target_time=float(t.item()),
                    scale_factor=scale_factor,
                    reference_eigenvectors=eigenvectors,
                )[0]
                transformed = self._denormalize_image(integrated).clamp(0.0, 1.0)
                row_images.append(transformed.cpu())
            rows.append(row_images)

        canvas = self._build_preview_canvas(rows)
        caption = (
            f"rows=tracked metric eigenvectors 0..{num_fields - 1}, "
            f"cols=t in [{time_values[0].item():.2f}, {time_values[-1].item():.2f}], "
            f"epsilon={epsilon_value:.4g}, rk4_substeps={self.config.preview_rk4_substeps}, "
            f"initial_eigenvalues={[round(v.item(), 4) for v in eigenvalues[:num_fields]]}"
        )

        experiment = self.logger.experiment
        if hasattr(experiment, "log"):
            import wandb

            experiment.log(
                {
                    "val/vector_field_integrations": wandb.Image(canvas, caption=caption),
                },
                step=self.global_step,
            )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._log_vector_field_grid()
        self._log_vector_field_preview()

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
