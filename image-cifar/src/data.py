import os
import torch_xla
import torch
import torch_xla.core.xla_model as xm
import torch.distributed as dist
import datasets
from enum import Enum
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from typing import Final, Dict, Tuple


class Dataset(str, Enum):
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"


DATASET_SOURCES: Final[Dict[Dataset, str]] = {
    Dataset.CIFAR10: "uoft-cs/cifar10",
    Dataset.CIFAR100: "uoft-cs/cifar100",
}


IMAGE_NAMES: Final[Dict[Dataset, str]] = {
    Dataset.CIFAR10: "img",
    Dataset.CIFAR100: "img",
}


LABEL_NAMES: Final[Dict[Dataset, str]] = {
    Dataset.CIFAR10: "label",
    Dataset.CIFAR100: "fine_label",
}


def batch_random_crop_flip(
    images: torch.Tensor,
    generator: torch.Generator,
    padding: int = 4,
    crop_size: int = 32,
    flip_p: float = 0.5,
) -> torch.Tensor:
    B, C, H, W = images.shape
    padded_images = F.pad(images, (padding, padding, padding, padding))

    x_shifts = torch.randint(
        0,
        H + 2 * padding - crop_size + 1,
        (B,),
        device=images.device,
        generator=generator,
    )
    y_shifts = torch.randint(
        0,
        W + 2 * padding - crop_size + 1,
        (B,),
        device=images.device,
        generator=generator,
    )
    flip_mask = torch.rand(B, device=images.device, generator=generator) < flip_p

    augmented_images = torch.empty((B, C, crop_size, crop_size), device=images.device, dtype=images.dtype)
    tmp_images = torch.empty((B, C, crop_size, W + 2 * padding), device=images.device, dtype=images.dtype)

    for i in range(H + 2 * padding - crop_size + 1):
        mask = x_shifts == i
        if mask.any():
            tmp_images[mask] = padded_images[mask, :, i : i + crop_size, :]
    for j in range(W + 2 * padding - crop_size + 1):
        mask = (y_shifts == j) & (~flip_mask)
        if mask.any():
            augmented_images[mask] = tmp_images[mask, :, :, j : j + crop_size]
        mask = (y_shifts == j) & (flip_mask)
        if mask.any():
            augmented_images[mask] = torch.flip(tmp_images[mask, :, :, j : j + crop_size], dims=[3])
    return augmented_images


class CifarLoader:
    def __init__(
        self,
        ds_name: Dataset = Dataset.CIFAR10,
        train: bool = True,
        batch_size: int = 1024,
        rank: int = 0,
        num_replicas: int = 1,
        base_seed: int = 42,
        data_path: str = "./data",
    ):
        if not xm.is_master_ordinal(local=True):
            xm.xla_rendezvous(b"data_loader_init")
        
        device = torch_xla.device()

        # default path to store processed datasets
        pt_path = os.path.join(data_path, ds_name + ("_train.pt" if train else "_test.pt"))

        if not os.path.exists(pt_path):
            dataset = datasets.load_dataset(
                DATASET_SOURCES[ds_name],
                split="train" if train else "test",
                cache_dir=os.path.join(data_path, "cache"),
            )
            dataset = dataset.with_format("torch")

            # extract images and labels and save as torch tensors
            images = dataset[:][IMAGE_NAMES[ds_name]].to(device).detach().clone()
            labels = dataset[:][LABEL_NAMES[ds_name]].to(device).detach().clone()

            # compute mean and std for normalization
            mean = (images.float().mean(dim=[0, 2, 3]) / 255.0).tolist()
            std = (images.float().std(dim=[0, 2, 3]) / 255.0).tolist()

            # save processed dataset
            torch.save(
                {
                    "images": images,
                    "labels": labels,
                    "mean": mean,
                    "std": std,
                    "num_classes": int(labels.max().item() + 1),
                },
                pt_path,
            )
        
        if xm.is_master_ordinal(local=True):
            xm.xla_rendezvous(b"data_loader_init")

        # load processed dataset
        data = torch.load(pt_path, map_location=device)
        self._images, self._labels, mean, std, self._num_classes = (
            data["images"],
            data["labels"],
            data["mean"],
            data["std"],
            data["num_classes"],
        )

        # setup parameters
        self._batch_size = batch_size
        self._rank = rank
        self._num_replicas = num_replicas
        assert (self._num_replicas == 0) or (self._batch_size % self._num_replicas == 0), (
            "Batch size must be divisible by num_replicas"
        )
        self._base_seed = base_seed
        assert self._base_seed >= 0, "Base seed must be non-negative"
        self._drop_last = train
        self._shuffle = train
        self._epoch = 0

        # define transformations
        self._transform = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=mean, std=std),
            ]
        )

        if train:
            self._batch_transform = lambda images, generator: self._transform(
                batch_random_crop_flip(images, generator, padding=4, crop_size=32, flip_p=0.5)
            )
        else:
            self._batch_transform = lambda images, _: self._transform(images)

    def __len__(self) -> int:
        return (
            (len(self._images) + self._batch_size - 1) // self._batch_size
            if not self._drop_last
            else len(self._images) // self._batch_size
        )

    def __iter__(self):
        augmented_images, indices = self._get_augmented_images_and_indices()

        for i in range(len(self)):
            idx = indices[i * self._batch_size : min((i + 1) * self._batch_size, self._images.shape[0])]
            idx = idx[self._rank :: self._num_replicas] if self._num_replicas > 0 else idx
            if idx.numel() > 0:
                if self._num_replicas > 0:
                    yield (
                        augmented_images[idx].view(-1, *augmented_images.shape[1:]),
                        self._labels[idx].view(-1),
                    )
                else:
                    yield augmented_images[idx], self._labels[idx]

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _get_augmented_images_and_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=self._images.device)
        generator.manual_seed(self._epoch + self._base_seed * 1007)
        augmented_images = self._batch_transform(self._images, generator)
        if self._shuffle:
            indices = torch.randperm(len(augmented_images), device=self._images.device, generator=generator)
        else:
            indices = torch.arange(len(augmented_images), device=self._images.device)
        return augmented_images, indices
