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
    tikhonov_lambda: float = 1e-4
    preview_fields: int = 8
    preview_samples: int = 4
    preview_steps: int = 7
    preview_scale: float = 0.25
    preview_rk4_substeps: int = 8


class MetricMatchingModule(L.LightningModule):
    def __init__(self, config: MetricMatchingConfig) -> None:
        super().__init__()
        self.config = config
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

    def forward(self, image: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return self.network(image, epsilon)

    def sample_epsilon(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps_min = torch.tensor(self.config.epsilon_min, device=device).log()
        eps_max = torch.tensor(self.config.epsilon_max, device=device).log()
        return torch.exp(torch.rand(batch_size, device=device) * (eps_max - eps_min) + eps_min)

    def compute_low_rank_loss(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = images.shape[0]
        epsilon = self.sample_epsilon(batch_size, images.device)
        noise = torch.randn_like(images) * epsilon.sqrt()[:, None, None, None]
        noisy_images = images + noise
        delta = images - noisy_images

        metric_factors = self(noisy_images, epsilon)
        metric_flat = metric_factors.flatten(start_dim=2)
        delta_flat = delta.flatten(start_dim=1)
        rank = metric_flat.shape[1]
        data_dim = metric_flat.shape[2]
        normalization = float(data_dim**2)

        gram = torch.matmul(metric_flat, metric_flat.transpose(1, 2))
        frob_term = (gram**2).sum(dim=(1, 2)) / normalization
        projected_delta = torch.matmul(metric_flat, delta_flat.unsqueeze(-1)).squeeze(-1)
        alignment_term = projected_delta.pow(2).sum(dim=1) / (epsilon * normalization)
        # For the conditional target Delta Delta^T / (2 epsilon), the omitted
        # constant is ||Delta||^4 / (4 epsilon^2). Adding it keeps gradients
        # unchanged while making the minimum easier to interpret.
        target_sq_norm = delta_flat.pow(2).sum(dim=1)
        target_term = target_sq_norm.pow(2) / (4.0 * epsilon.pow(2) * normalization)
        tangent_dim = metric_flat.pow(2).sum(dim=(1, 2))
        reg_term = self.config.tikhonov_lambda * 2.0 * tangent_dim / normalization

        loss = (frob_term + reg_term - alignment_term).mean()
        target_norm = (target_sq_norm / data_dim).mean()
        factor_norm = metric_flat.pow(2).sum(dim=(1, 2)).div(rank * data_dim).mean()

        metrics = {
            "frob_term": frob_term.mean().detach(),
            "alignment_term": alignment_term.mean().detach(),
            "target_term": target_term.mean().detach(),
            "reg_term": reg_term.mean().detach(),
            "tangent_dim": tangent_dim.mean().detach(),
            "epsilon_mean": epsilon.mean().detach(),
            "target_delta_norm": target_norm.detach(),
            "factor_norm": factor_norm.detach(),
            "rank": torch.tensor(float(rank), device=images.device),
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
        basis_fields = self(normalized_image, epsilon)
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
        eigenvalues = singular_values.pow(2)
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
            basis_fields = self(normalized_batch, epsilon)
            eigenvectors, eigenvalues = self._top_metric_eigenvectors(basis_fields)

        num_fields = min(self.config.preview_fields, eigenvectors.shape[1])
        rows: list[list[torch.Tensor]] = []
        rows.append([self._denormalize_image(image).clamp(0.0, 1.0).cpu() for image in normalized_batch])

        displayed_eigenvalues = []
        for field_idx in range(num_fields):
            row_fields = eigenvectors[:, field_idx]
            row_scale = row_fields.pow(2).mean(dim=(1, 2, 3)).sqrt().max().clamp_min(1e-6)
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
            rms = vector_field.pow(2).mean().sqrt().clamp_min(1e-6)
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
