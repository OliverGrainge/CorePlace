import os
from collections import defaultdict
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.train import CorePlaceDataset
from dataloaders.val import load_val_dataset


class CorePlaceDataModule(LightningDataModule):
    def __init__(
        self,
        dataconfig: Union[pd.DataFrame, str],
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 0,
        num_images_per_place: int = 100,
        val_dataset_names: List[str] = None,
    ):
        super().__init__()
        self.dataconfig = self._load_dataconfig(dataconfig)
        # Default transform if none provided - converts PIL to tensor
        self.transform = self._default_transform(image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_images_per_place = num_images_per_place
        self.val_dataset_names = val_dataset_names
        self.val_datasets = [
            load_val_dataset(name, self.transform) for name in self.val_dataset_names
        ]

    def _default_transform(self, image_size: int = 256):
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                #),
            ]
        )

    @classmethod
    def from_config(cls, config: dict):
        dataconfig = config.get("dataconfig")
        image_size = config.get("image_size", 256)
        batch_size = config.get("batch_size", 32)
        num_workers = config.get("num_workers", 0)
        num_images_per_place = config.get("num_images_per_place", 100)
        val_dataset_names = config.get("val_dataset_names", None)
        return cls(
            dataconfig=dataconfig,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            num_images_per_place=num_images_per_place,
            val_dataset_names=val_dataset_names,
        )

    def _load_dataconfig(self, dataconfig):
        if isinstance(dataconfig, str):
            assert os.path.exists(
                dataconfig
            ), f"Dataconfig file {dataconfig} does not exist"
            return pd.read_pickle(dataconfig)
        return dataconfig

    def setup(self, stage: Optional[str] = None):

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

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size * self.num_images_per_place,
                num_workers=self.num_workers,
            )
            for dataset in self.val_datasets
        ]

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

    dataloader = CorePlaceDataModule.from_config(
        {
            "dataconfig": "registry/coreplacesets/baseline.pkl",
            "num_images_per_place": 5,
            "val_dataset_names": ["Amstertime"],
            "batch_size": 4,
        }
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
    os.makedirs("tmp", exist_ok=True)
    plt.savefig("tmp/train_dataloader.png")

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
    test_datamodule("registry/coreplacesets/baseline.pkl")
