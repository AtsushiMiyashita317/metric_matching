from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split

from differentiable_3dshapes import Differentiable3Dshapes


COLOR_FACTOR_INDICES = (0, 1, 2)


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


@dataclass(frozen=True)
class Shapes3DStorage:
    image_key: str
    label_key: Optional[str]
    images: np.ndarray
    images_chw: np.ndarray
    labels: Optional[np.ndarray]
    factor_values: Optional[tuple[np.ndarray, ...]]
    label_indices: Optional[np.ndarray]
    lookup_shape: Optional[tuple[int, ...]]
    factor_strides: Optional[np.ndarray]
    color_neighbor_grid: Optional[np.ndarray]
    height: int
    width: int
    channels: int
    length: int


@lru_cache(maxsize=4)
def _load_storage(h5_path: str) -> Shapes3DStorage:
    with h5py.File(h5_path, "r") as h5_file:
        image_key, label_key = _resolve_dataset_keys(h5_file)
        images = np.array(h5_file[image_key], dtype=np.uint8, copy=True)
        labels = (
            np.array(h5_file[label_key], dtype=np.float32, copy=True)
            if label_key is not None
            else None
        )

    image_shape = images.shape[1:]
    if len(image_shape) != 3:
        raise ValueError(f"Expected image tensors with shape (H, W, C), got {image_shape}")
    height, width, channels = (int(dim) for dim in image_shape)

    factor_values = None
    label_indices = None
    lookup_shape = None
    factor_strides = None
    color_neighbor_grid = None
    if labels is not None:
        factor_values = tuple(np.unique(labels[:, dim]).astype(np.float32) for dim in range(labels.shape[1]))
        label_indices = np.stack(
            [np.searchsorted(values, labels[:, dim]) for dim, values in enumerate(factor_values)],
            axis=1,
        ).astype(np.int64, copy=False)
        lookup_shape = tuple(len(values) for values in factor_values)
        factor_strides = np.array(
            [int(np.prod(lookup_shape[dim + 1 :], dtype=np.int64)) for dim in range(len(lookup_shape))],
            dtype=np.int64,
        )
        direct_indices = np.sum(label_indices * factor_strides[None, :], axis=1, dtype=np.int64)
        expected_indices = np.arange(labels.shape[0], dtype=np.int64)
        if not np.array_equal(direct_indices, expected_indices):
            raise ValueError("Dataset ordering does not match the expected factor-major 3dshapes layout.")

        offsets = (
            np.stack(
                np.meshgrid(
                    np.arange(3, dtype=np.int64),
                    np.arange(3, dtype=np.int64),
                    np.arange(3, dtype=np.int64),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
            - 1
        )
        base_color_indices = label_indices[:, None, :3]
        wrapped_color_indices = (base_color_indices + offsets[None, :, :]) % np.array(
            lookup_shape[:3],
            dtype=np.int64,
        )[None, None, :]
        color_index_deltas = wrapped_color_indices - base_color_indices
        color_neighbor_grid = (
            expected_indices[:, None]
            + np.sum(color_index_deltas * factor_strides[None, None, :3], axis=2, dtype=np.int64)
        ).reshape(-1, 3, 3, 3)

    return Shapes3DStorage(
        image_key=image_key,
        label_key=label_key,
        images=images,
        images_chw=np.transpose(images, (0, 3, 1, 2)),
        labels=labels,
        factor_values=factor_values,
        label_indices=label_indices,
        lookup_shape=lookup_shape,
        factor_strides=factor_strides,
        color_neighbor_grid=color_neighbor_grid,
        height=height,
        width=width,
        channels=channels,
        length=int(images.shape[0]),
    )


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


def apply_gaussian_smoothing_with_kernel(
    image: torch.Tensor,
    kernel: Optional[torch.Tensor],
) -> torch.Tensor:
    if kernel is None:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")
    radius = kernel.shape[-1] // 2
    padded = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode="reflect")
    smoothed = F.conv2d(padded, kernel, groups=image.shape[0])
    return smoothed.squeeze(0)


def restore_image_range(
    image: torch.Tensor,
    stats: Optional[DatasetStats] = None,
) -> torch.Tensor:
    if stats is None:
        return (image + 1.0) / 2.0
    mean = stats.mean.to(device=image.device, dtype=image.dtype)
    std = stats.std.to(device=image.device, dtype=image.dtype)
    return image * std + mean


class Shapes3DDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        normalize: bool = True,
        stats: Optional[DatasetStats] = None,
        smoothing_sigma: float = 0.6,
        indices: Optional[np.ndarray] = None,
        enable_color_interpolation: bool = False,
    ) -> None:
        self.h5_path = str(Path(h5_path).resolve())
        self.normalize = normalize
        self.stats = stats
        self.smoothing_sigma = smoothing_sigma
        self.enable_color_interpolation = enable_color_interpolation
        if self.smoothing_sigma < 0.0:
            raise ValueError(f"smoothing_sigma must be non-negative, got {self.smoothing_sigma}")
        self.storage = _load_storage(self.h5_path)
        self.image_key = self.storage.image_key
        self.label_key = self.storage.label_key
        self.images = self.storage.images
        self.images_chw = self.storage.images_chw
        self.labels = self.storage.labels
        self.height = self.storage.height
        self.width = self.storage.width
        self.channels = self.storage.channels
        self._image_scale = 1.0 / 255.0
        self._smoothing_kernel = (
            _gaussian_kernel_2d(
                sigma=self.smoothing_sigma,
                channels=self.channels,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            if self.smoothing_sigma > 0.0
            else None
        )
        self.base_indices = (
            np.arange(self.storage.length, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.base_length = int(self.base_indices.shape[0])
        self.length = self.base_length
        if self.enable_color_interpolation and self.labels is None:
            raise ValueError("Color interpolation requires labels to be present in the dataset.")

    def __len__(self) -> int:
        return self.length

    def _image_to_tensor(self, image_chw: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image_chw).to(dtype=torch.float32).mul_(self._image_scale)
        tensor = apply_gaussian_smoothing_with_kernel(tensor, kernel=self._smoothing_kernel)
        if self.normalize and self.stats is not None:
            tensor = (tensor - self.stats.mean) / self.stats.std
        elif not self.normalize:
            tensor = tensor.mul(2.0).sub(1.0)
        return tensor

    def _sample_alpha(self) -> float:
        return float(torch.rand((), dtype=torch.float32).item())

    def _lookup_index_from_factor_indices(self, factor_indices: np.ndarray) -> int:
        if self.storage.factor_strides is None:
            raise RuntimeError("Label lookup tables are unavailable for interpolation.")
        return int(np.dot(factor_indices.astype(np.int64, copy=False), self.storage.factor_strides))

    def _interpolate_wrapped_color_value(
        self,
        factor_index: int,
        lower_factor_index: int,
        alpha: float,
    ) -> float:
        if self.storage.factor_values is None:
            raise RuntimeError("Factor values are unavailable for interpolation.")
        values = self.storage.factor_values[factor_index]
        if len(values) == 1:
            return float(values[0])
        step = float(values[1] - values[0])
        wrapped_position = (lower_factor_index + alpha) % len(values)
        return float(values[0] + step * wrapped_position)

    def _build_interpolated_sample(self, base_index: int) -> dict[str, torch.Tensor]:
        if self.storage.label_indices is None or self.storage.color_neighbor_grid is None:
            raise RuntimeError("Interpolation lookup tables are required for interpolation.")

        base_factor_indices = self.storage.label_indices[base_index]
        alphas = np.zeros(len(COLOR_FACTOR_INDICES), dtype=np.float32)
        direction_starts = np.empty(len(COLOR_FACTOR_INDICES), dtype=np.int64)
        for slot in range(len(COLOR_FACTOR_INDICES)):
            alphas[slot] = self._sample_alpha()
            direction_starts[slot] = int(torch.randint(2, size=(1,)).item())

        neighbor_grid = self.storage.color_neighbor_grid[base_index]
        subcube = neighbor_grid[
            direction_starts[0] : direction_starts[0] + 2,
            direction_starts[1] : direction_starts[1] + 2,
            direction_starts[2] : direction_starts[2] + 2,
        ]

        axis_weights = (
            np.array([1.0 - alphas[0], alphas[0]], dtype=np.float32),
            np.array([1.0 - alphas[1], alphas[1]], dtype=np.float32),
            np.array([1.0 - alphas[2], alphas[2]], dtype=np.float32),
        )
        weights = (
            axis_weights[0][:, None, None]
            * axis_weights[1][None, :, None]
            * axis_weights[2][None, None, :]
        ).reshape(-1)
        support_indices = subcube.reshape(-1)

        support_images = self.images_chw[support_indices].astype(np.float32, copy=False)
        mixed_image = np.tensordot(weights, support_images, axes=(0, 0))

        mixed_label = None
        if self.labels is not None:
            mixed_label = self.labels[base_index].copy()
            for slot, factor_index in enumerate(COLOR_FACTOR_INDICES):
                lower_factor_index = int((base_factor_indices[factor_index] + direction_starts[slot] - 1))
                mixed_label[factor_index] = self._interpolate_wrapped_color_value(
                    factor_index=factor_index,
                    lower_factor_index=lower_factor_index,
                    alpha=float(alphas[slot]),
                )

        image = self._image_to_tensor(mixed_image)
        target_corner_index = 7
        target_index = int(support_indices[target_corner_index])

        sample = {
            "image": image,
            "is_interpolated": torch.tensor(True),
            "interpolation_alpha": torch.tensor(float(alphas.mean()), dtype=torch.float32),
            "interpolation_alphas": torch.from_numpy(alphas.copy()),
            "interpolation_factor": torch.tensor(-2, dtype=torch.int64),
            "source_index": torch.tensor(base_index, dtype=torch.int64),
            "target_index": torch.tensor(target_index, dtype=torch.int64),
        }
        if mixed_label is not None:
            sample["label"] = torch.from_numpy(mixed_label.astype(np.float32, copy=False))
        return sample

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} is out of range for dataset of length {self.length}.")

        base_index = int(self.base_indices[index])
        if self.enable_color_interpolation:
            return self._build_interpolated_sample(base_index)

        image = self._image_to_tensor(self.images_chw[base_index])
        sample = {
            "image": image,
            "is_interpolated": torch.tensor(False),
            "interpolation_alpha": torch.tensor(0.0, dtype=torch.float32),
            "interpolation_alphas": torch.zeros(len(COLOR_FACTOR_INDICES), dtype=torch.float32),
            "interpolation_factor": torch.tensor(-1, dtype=torch.int64),
            "source_index": torch.tensor(base_index, dtype=torch.int64),
            "target_index": torch.tensor(base_index, dtype=torch.int64),
        }
        if self.labels is not None:
            sample["label"] = torch.from_numpy(self.labels[base_index].copy())
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
        enable_color_interpolation: bool = False,
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
        self.enable_color_interpolation = enable_color_interpolation
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
            enable_color_interpolation=False,
        )
        self.image_shape = (
            base_dataset.channels,
            base_dataset.height,
            base_dataset.width,
        )

        if self.normalize:
            self.stats = self._compute_stats(base_dataset)

        if base_dataset.label_key is not None and base_dataset.labels is not None:
            self.factor_dim = int(base_dataset.labels.shape[-1])

        total_base_length = base_dataset.base_length
        val_length = max(1, int(total_base_length * self.val_fraction))
        train_length = total_base_length - val_length
        train_subset, val_subset = random_split(
            range(total_base_length),
            [train_length, val_length],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_indices = base_dataset.base_indices[np.array(train_subset.indices, dtype=np.int64)]
        val_indices = base_dataset.base_indices[np.array(val_subset.indices, dtype=np.int64)]

        if self.max_train_samples is not None:
            train_indices = train_indices[: min(self.max_train_samples, len(train_indices))]
        if self.max_val_samples is not None:
            val_indices = val_indices[: min(self.max_val_samples, len(val_indices))]

        train_dataset = Shapes3DDataset(
            self.data_path,
            normalize=self.normalize,
            stats=self.stats,
            smoothing_sigma=self.smoothing_sigma,
            indices=train_indices,
            enable_color_interpolation=self.enable_color_interpolation,
        )
        val_dataset = Shapes3DDataset(
            self.data_path,
            normalize=self.normalize,
            stats=self.stats,
            smoothing_sigma=self.smoothing_sigma,
            indices=val_indices,
            enable_color_interpolation=False,
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


class Shapes3DFactorDataset(Dataset):
    def __init__(
        self,
        length: int,
        normalize: bool = True,
        stats: Optional[DatasetStats] = None,
        seed: int = 42,
        indices: Optional[np.ndarray] = None,
        random_sampling: bool = False,
        return_grad: bool = False,
    ) -> None:
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        self.length = int(length)
        self.normalize = normalize
        self.stats = stats
        self.seed = int(seed)
        self.random_sampling = random_sampling
        self.return_grad = return_grad
        self.indices = (
            np.arange(self.length, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.factor_dim = 6
        self._rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _sample_factors(self, global_index: int) -> torch.Tensor:
        if self.random_sampling:
            if self._rng is None:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None:
                    worker_seed = self.seed
                else:
                    # Prefer DataLoader-provided worker seed so sequences can vary per epoch.
                    worker_seed = int(worker_info.seed) + self.seed
                self._rng = np.random.default_rng(worker_seed)
            rng = self._rng
        else:
            rng = np.random.default_rng(self.seed + int(global_index))
        shape = float(rng.integers(0, 4))
        size = float(rng.uniform(0.7, 1.5))
        orientation = float(rng.uniform(0.0, 1.0))
        floor_hue = float(rng.uniform(0.0, 1.0))
        wall_hue = float(rng.uniform(0.0, 1.0))
        object_hue = float(rng.uniform(0.0, 1.0))
        return torch.tensor(
            [shape, size, orientation, floor_hue, wall_hue, object_hue],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}.")

        global_index = int(self.indices[index])
        factors = self._sample_factors(global_index)

        sample = {
            "label": factors,
            "is_interpolated": torch.tensor(False),
            "interpolation_alpha": torch.tensor(0.0, dtype=torch.float32),
            "interpolation_alphas": torch.zeros(len(COLOR_FACTOR_INDICES), dtype=torch.float32),
            "interpolation_factor": torch.tensor(-1, dtype=torch.int64),
            "source_index": torch.tensor(global_index, dtype=torch.int64),
            "target_index": torch.tensor(global_index, dtype=torch.int64),
            "return_grad": torch.tensor(self.return_grad),
        }
        return sample


class Shapes3DInfiniteFactorDataset(IterableDataset):
    def __init__(
        self,
        normalize: bool = True,
        stats: Optional[DatasetStats] = None,
        seed: int = 42,
    ) -> None:
        self.normalize = normalize
        self.stats = stats
        self.seed = int(seed)
        self.factor_dim = 6

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + worker_id * 100_003)
        running_index = worker_id
        while True:
            shape = float(rng.integers(0, 4))
            size = float(rng.uniform(0.7, 1.5))
            orientation = float(rng.uniform(0.0, 1.0))
            floor_hue = float(rng.uniform(0.0, 1.0))
            wall_hue = float(rng.uniform(0.0, 1.0))
            object_hue = float(rng.uniform(0.0, 1.0))
            factors = torch.tensor(
                [shape, size, orientation, floor_hue, wall_hue, object_hue],
                dtype=torch.float32,
            )

            yield {
                "label": factors,
                "is_interpolated": torch.tensor(False),
                "interpolation_alpha": torch.tensor(0.0, dtype=torch.float32),
                "interpolation_alphas": torch.zeros(len(COLOR_FACTOR_INDICES), dtype=torch.float32),
                "interpolation_factor": torch.tensor(-1, dtype=torch.int64),
                "source_index": torch.tensor(running_index, dtype=torch.int64),
                "target_index": torch.tensor(running_index, dtype=torch.int64),
                "return_grad": torch.tensor(False),
            }
            running_index += max(1, (worker_info.num_workers if worker_info is not None else 1))


class Differentiable3DshapesDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        normalize: bool = True,
        stats_samples: int = 2048,
        total_samples: int = 100_000,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        seed: int = 42,
        random_sampling: bool = True,
        infinite_train_stream: bool = False,
        val_return_grad: bool = False,
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("Differentiable3DshapesDataModule requires CUDA for batched GPU rendering.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.normalize = normalize
        self.stats_samples = stats_samples
        self.total_samples = total_samples
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.seed = seed
        self.random_sampling = random_sampling
        self.infinite_train_stream = infinite_train_stream
        self.val_return_grad = val_return_grad
        self.stats: Optional[DatasetStats] = None
        self.train_dataset: Optional[Dataset | IterableDataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.image_shape: Optional[tuple[int, int, int]] = None
        self.factor_dim: Optional[int] = 6
        self._render_device = torch.device("cuda")
        self._gpu_renderer: Optional[Differentiable3Dshapes] = None

    def _get_gpu_renderer(self) -> Differentiable3Dshapes:
        if self._gpu_renderer is None:
            self._gpu_renderer = Differentiable3Dshapes()
        return self._gpu_renderer

    def _render_from_factors(
        self,
        factors: torch.Tensor,
        return_grad: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        renderer = self._get_gpu_renderer()
        factors = factors.to(device=device, dtype=torch.float32, non_blocking=True)
        shape = factors[:, 0].to(dtype=torch.int64)
        size = factors[:, 1]
        orientation = factors[:, 2]
        floor_hue = factors[:, 3]
        wall_hue = factors[:, 4]
        object_hue = factors[:, 5]

        if return_grad:
            grads, images = renderer.forward(
                shape=shape,
                size=size,
                orientation=orientation,
                floor_hue=floor_hue,
                wall_hue=wall_hue,
                object_hue=object_hue,
                return_grad=True,
            )
            images = images.to(dtype=torch.float32)
            gt_tangent = torch.stack([grad.to(dtype=torch.float32) for grad in grads], dim=1)
            return images, gt_tangent

        images = renderer.forward(
            shape=shape,
            size=size,
            orientation=orientation,
            floor_hue=floor_hue,
            wall_hue=wall_hue,
            object_hue=object_hue,
        ).to(dtype=torch.float32)
        return images, None

    def _apply_normalization(
        self,
        image: torch.Tensor,
        gt_tangent: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.normalize and self.stats is not None:
            mean = self.stats.mean.to(device=image.device, dtype=image.dtype)
            std = self.stats.std.to(device=image.device, dtype=image.dtype)
            image = (image - mean) / std
            if gt_tangent is not None:
                gt_tangent = gt_tangent / std.unsqueeze(0).unsqueeze(0)
        elif not self.normalize:
            image = image.mul(2.0).sub(1.0)
            if gt_tangent is not None:
                gt_tangent = gt_tangent.mul(2.0)
        return image, gt_tangent

    def prepare_data(self) -> None:
        return

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        probe, _ = self._render_from_factors(
            factors=torch.tensor(
                [[0.0, 1.0, 0.0, 0.0, 0.33, 0.66]],
                dtype=torch.float32,
            ),
            return_grad=False,
            device=self._render_device,
        )
        probe = probe[0]
        if probe.ndim != 3:
            raise ValueError(f"Expected renderer output shape (C,H,W), got {tuple(probe.shape)}")
        self.image_shape = (int(probe.shape[0]), int(probe.shape[1]), int(probe.shape[2]))

        val_length = max(1, int(self.total_samples * self.val_fraction))
        train_length = self.total_samples - val_length
        if train_length <= 0:
            raise ValueError(
                f"total_samples={self.total_samples} and val_fraction={self.val_fraction} "
                "produce an empty train split."
            )

        all_indices = np.arange(self.total_samples, dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_indices)
        train_indices = all_indices[:train_length]
        val_indices = all_indices[train_length:]

        if self.max_train_samples is not None:
            train_indices = train_indices[: min(self.max_train_samples, len(train_indices))]
        if self.max_val_samples is not None:
            val_indices = val_indices[: min(self.max_val_samples, len(val_indices))]

        raw_train_dataset = Shapes3DFactorDataset(
            length=self.total_samples,
            normalize=False,
            stats=None,
            seed=self.seed,
            indices=train_indices,
            random_sampling=self.random_sampling,
            return_grad=False,
        )
        if self.normalize:
            self.stats = self._compute_stats(raw_train_dataset)

        if self.infinite_train_stream:
            self.train_dataset = Shapes3DInfiniteFactorDataset(
                normalize=self.normalize,
                stats=self.stats,
                seed=self.seed,
            )
        else:
            self.train_dataset = Shapes3DFactorDataset(
                length=self.total_samples,
                normalize=self.normalize,
                stats=self.stats,
                seed=self.seed,
                indices=train_indices,
                random_sampling=self.random_sampling,
                return_grad=False,
            )
        self.val_dataset = Shapes3DFactorDataset(
            length=self.total_samples,
            normalize=self.normalize,
            stats=self.stats,
            seed=self.seed,
            indices=val_indices,
            random_sampling=False,
            return_grad=self.val_return_grad,
        )

    def _compute_stats(self, dataset: Shapes3DFactorDataset) -> DatasetStats:
        if self.stats_samples is not None and self.stats_samples < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(self.stats_samples))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        channel_sum = None
        channel_sq_sum = None
        pixel_count = 0
        for batch in loader:
            images, _ = self._render_from_factors(
                factors=batch["label"],
                return_grad=False,
                device=self._render_device,
            )
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
        return DatasetStats(
            mean=mean[:, None, None].to("cpu"),
            std=std[:, None, None].to("cpu"),
        )

    def on_after_batch_transfer(
        self,
        batch: object,
        dataloader_idx: int,
    ) -> object:
        if not isinstance(batch, dict) or "label" not in batch:
            return batch
        factors = batch["label"]
        return_grad = bool(batch["return_grad"].any().item()) if "return_grad" in batch else False
        image, gt_tangent = self._render_from_factors(
            factors=factors,
            return_grad=return_grad,
            device=self._render_device,
        )
        image, gt_tangent = self._apply_normalization(image=image, gt_tangent=gt_tangent)
        batch["image"] = image
        if gt_tangent is not None:
            batch["gt_tangent"] = gt_tangent
        if "return_grad" in batch:
            batch.pop("return_grad")
        return batch

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.infinite_train_stream,
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
