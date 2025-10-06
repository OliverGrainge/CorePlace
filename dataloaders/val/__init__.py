import os
from typing import Union

from torchvision import transforms

from dataloaders.val.valdataset import ValDataset


def load_val_dataset(
    val_dataset_name: str, transform: Union[transforms.Compose, None] = None
):
    valconfig_root = os.path.join(os.path.dirname(__file__), "valconfigs")
    available_configs = os.listdir(valconfig_root)
    available_configs = [cfg.replace(".pkl", "") for cfg in available_configs]

    assert (
        val_dataset_name in available_configs
    ), f"Val dataset {val_dataset_name} not supported, only {available_configs} are available"
    return ValDataset(
        root=valconfig_root, valconfig_name=val_dataset_name, transform=transform
    )
