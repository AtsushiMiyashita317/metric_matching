from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


def _resolve_dataset_keys(h5_file: h5py.File) -> tuple[str, Optional[str]]:
    image_key = None
    label_key = None
    for key in h5_file.keys():
        item = h5_file[key]
        if not isinstance(item, h5py.Dataset):
            continue
        if item.ndim >= 3 and image_key is None:
            image_key = key
        elif item.ndim == 2 and label_key is None:
            label_key = key
    if image_key is None:
        raise ValueError("Could not find an image dataset inside the HDF5 file.")
    return image_key, label_key


@dataclass
class DatasetStats:
    mean: torch.Tensor
    std: torch.Tensor


def _gaussian_kernel_2d(
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = max(1, int(ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, -1, -1).contiguous()


def apply_gaussian_smoothing(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")

    kernel = _gaussian_kernel_2d(
        sigma=sigma,
        channels=image.shape[0],
        device=image.device,
        dtype=image.dtype,
    )
    radius = kernel.shape[-1] // 2
    padded = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode="reflect")
    smoothed = F.conv2d(padded, kernel, groups=image.shape[0])
    return smoothed.squeeze(0)


class Shapes3DDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        normalize: bool = True,
        stats: Optional[DatasetStats] = None,
        smoothing_sigma: float = 0.6,
    ) -> None:
        self.h5_path = str(h5_path)
        self.normalize = normalize
        self.stats = stats
        self.smoothing_sigma = smoothing_sigma
        if self.smoothing_sigma < 0.0:
            raise ValueError(f"smoothing_sigma must be non-negative, got {self.smoothing_sigma}")
        with h5py.File(self.h5_path, "r") as h5_file:
            image_key, label_key = _resolve_dataset_keys(h5_file)
            self.image_key = image_key
            self.label_key = label_key
            self.images = np.array(h5_file[self.image_key], dtype=np.uint8, copy=True)
            self.labels = (
                np.array(h5_file[self.label_key], dtype=np.float32, copy=True)
                if self.label_key is not None
                else None
            )
            self.length = int(self.images.shape[0])
            image_shape = self.images.shape[1:]
        if len(image_shape) != 3:
            raise ValueError(f"Expected image tensors with shape (H, W, C), got {image_shape}")
        self.height, self.width, self.channels = image_shape

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.images[index]
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
        image = apply_gaussian_smoothing(image, sigma=self.smoothing_sigma)
        if self.normalize and self.stats is not None:
            image = (image - self.stats.mean) / self.stats.std

        sample = {"image": image}
        if self.labels is not None:
            sample["label"] = torch.from_numpy(self.labels[index])
        return sample


class Shapes3DDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str = "3dshapes.h5",
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        normalize: bool = True,
        stats_samples: int = 8192,
        smoothing_sigma: float = 0.6,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.normalize = normalize
        self.stats_samples = stats_samples
        self.smoothing_sigma = smoothing_sigma
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.seed = seed
        self.stats: Optional[DatasetStats] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.image_shape: Optional[tuple[int, int, int]] = None
        self.factor_dim: Optional[int] = None

    def prepare_data(self) -> None:
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        base_dataset = Shapes3DDataset(
            self.data_path,
            normalize=False,
            smoothing_sigma=self.smoothing_sigma,
        )
        self.image_shape = (
            base_dataset.channels,
            base_dataset.height,
            base_dataset.width,
        )

        if self.normalize:
            self.stats = self._compute_stats(base_dataset)

        full_dataset = Shapes3DDataset(
            self.data_path,
            normalize=self.normalize,
            stats=self.stats,
            smoothing_sigma=self.smoothing_sigma,
        )
        if full_dataset.label_key is not None:
            with h5py.File(self.data_path, "r") as h5_file:
                self.factor_dim = int(h5_file[full_dataset.label_key].shape[-1])

        total_length = len(full_dataset)
        val_length = max(1, int(total_length * self.val_fraction))
        train_length = total_length - val_length
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(self.seed),
        )

        if self.max_train_samples is not None:
            train_dataset = torch.utils.data.Subset(
                train_dataset,
                range(min(self.max_train_samples, len(train_dataset))),
            )
        if self.max_val_samples is not None:
            val_dataset = torch.utils.data.Subset(
                val_dataset,
                range(min(self.max_val_samples, len(val_dataset))),
            )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _compute_stats(self, dataset: Shapes3DDataset) -> DatasetStats:
        if self.stats_samples is not None and self.stats_samples < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(self.stats_samples))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        channel_sum = None
        channel_sq_sum = None
        pixel_count = 0
        for batch in loader:
            images = batch["image"]
            batch_sum = images.sum(dim=(0, 2, 3))
            batch_sq_sum = (images**2).sum(dim=(0, 2, 3))
            if channel_sum is None:
                channel_sum = batch_sum
                channel_sq_sum = batch_sq_sum
            else:
                channel_sum += batch_sum
                channel_sq_sum += batch_sq_sum
            pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

        mean = channel_sum / pixel_count
        var = channel_sq_sum / pixel_count - mean**2
        std = torch.sqrt(var.clamp_min(1e-6))
        return DatasetStats(mean=mean[:, None, None], std=std[:, None, None])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
