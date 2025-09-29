from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from typing import List, Tuple, Union, Optional
from torchvision import transforms


class CorePlaceDataset(Dataset):
    def __init__(
        self, dataconfig: pd.DataFrame, transform=None, num_images_per_place: int = 10
    ):
        self.dataconfig = dataconfig
        self.unique_ids = dataconfig["place_id"].unique()
        self.num_images_per_place = num_images_per_place
        self.transform = transform

        # Precompute indices for each place_id for efficient sampling
        self.placeid_to_indices = {
            place_id: dataconfig.index[dataconfig["place_id"] == place_id].tolist()
            for place_id in self.unique_ids
        }

    def __len__(self):
        return len(self.unique_ids)

    def __getitem__(self, index):
        place_id = self.unique_ids[index]
        indices = self.placeid_to_indices[place_id]
        n_images = len(indices)
        n_samples = self.num_images_per_place

        # Always sample exactly num_images_per_place, with replacement if needed
        replace = n_images < n_samples
        sampled_indices = np.random.choice(indices, size=n_samples, replace=replace)

        images = []
        for idx in sampled_indices:
            row = self.dataconfig.loc[idx]
            image_path = row["image_path"]
            img = Image.open(image_path)
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        labels = torch.full((n_samples,), place_id, dtype=torch.long)

        return images, labels


class CorePlaceDataModule(LightningDataModule):
    def __init__(
        self,
        dataconfig: Union[pd.DataFrame, str],
        transform=None,
        batch_size: int = 32,
        num_workers: int = 0,
        num_images_per_place: int = 100,
    ):
        super().__init__()
        self.dataconfig = dataconfig
        # Default transform if none provided - converts PIL to tensor
        self.transform = (
            transform if transform is not None else self._default_transform()
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_images_per_place = num_images_per_place

    def _default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @classmethod
    def from_dataconfig(
        cls,
        dataconfig: Union[pd.DataFrame, str],
        transform=None,
        batch_size: int = 32,
        num_workers: int = 0,
        num_images_per_place: int = 100,
    ):
        return cls(dataconfig, transform, batch_size, num_workers, num_images_per_place)

    def _load_dataconfig(self):
        if isinstance(self.dataconfig, str):
            assert os.path.exists(
                self.dataconfig
            ), f"Dataconfig file {self.dataconfig} does not exist"
            return pd.read_pickle(self.dataconfig)
        return self.dataconfig

    def setup(self, stage: Optional[str] = None):
        self.dataconfig = self._load_dataconfig()
        self.dataset = CorePlaceDataset(
            self.dataconfig,
            transform=self.transform,
            num_images_per_place=self.num_images_per_place,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        images, labels = zip(*batch)
        images = torch.vstack(images)
        labels = torch.hstack(labels)
        return images, labels


def test_datamodule(dataconfig_path: str):
    def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Denormalize a tensor image."""
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataloader = CorePlaceDataModule.from_dataconfig(
        dataconfig_path, batch_size=10, num_images_per_place=10
    )
    dataloader.setup()
    batch = next(iter(dataloader.train_dataloader()))

    # Organize images by class
    classes = defaultdict(list)
    for image, label in zip(batch[0], batch[1]):
        classes[label.item()].append(image)

    # Create visualization
    n_classes = len(classes)
    max_images_per_class = max(len(imgs) for imgs in classes.values())

    fig, axes = plt.subplots(
        n_classes,
        max_images_per_class,
        figsize=(max_images_per_class * 2, n_classes * 2),
    )

    # Handle single class case
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Images per Class", fontsize=16, fontweight="bold", y=0.995)

    for row_idx, (class_id, images) in enumerate(sorted(classes.items())):
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx] if n_classes > 1 else axes[col_idx]

            # Denormalize and display image
            img_denorm = denormalize(image)
            ax.imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")

            # Add class label on first image of each row
            if col_idx == 0:
                ax.set_ylabel(
                    f"Class {class_id}",
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    labelpad=40,
                    va="center",
                )

        # Hide empty subplots
        for col_idx in range(len(images), max_images_per_class):
            ax = axes[row_idx, col_idx] if n_classes > 1 else axes[col_idx]
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"Total images in batch: {len(batch[0])}")
    print(f"Number of classes: {n_classes}")
    print(f"\nImages per class:")
    for class_id, images in sorted(classes.items()):
        print(f"  Class {class_id}: {len(images)} images")
    print("=" * 50)


if __name__ == "__main__":
    test_datamodule("registry/datasets/test.pkl")
