import os
import pickle
from typing import Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ValDataset(Dataset):
    def __init__(
        self,
        root: str,
        valconfig_name: str,
        transform: Union[transforms.Compose, None] = None,
    ):
        self.path = os.path.join(root, valconfig_name + ".pkl")
        self.query, self.database, self.gt, self.dataset_folder = self._read_pkl()
        self.transform = (
            transform if transform is not None else self._default_transform()
        )
        self.n_query = len(self.query)
        self.n_database = len(self.database)
        self.images = self.query + self.database

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

    def _read_pkl(self):
        assert os.path.exists(self.path), f"Valconfig file {self.path} does not exist"
        assert self.path.endswith(
            ".pkl"
        ), f"Valconfig file {self.path} is not a .pkl file"
        with open(self.path, "rb") as f:
            valconfig = pickle.load(f)

        assert (
            "query" in valconfig.keys()
        ), f"Valconfig file {self.path} does not contain a query key"
        assert (
            "database" in valconfig.keys()
        ), f"Valconfig file {self.path} does not contain a database key"
        assert (
            "groundtruth" in valconfig.keys()
        ), f"Valconfig file {self.path} does not contain a groundtruth key"
        assert len(valconfig["query"]) == len(
            valconfig["groundtruth"]
        ), f"Valconfig file {self.path} does not have the same number of query andgroundtruth"
        assert (
            "dataset_folder" in valconfig.keys()
        ), f"Valconfig file {self.path} does not contain a dataset_folder key"
        return (
            valconfig["query"],
            valconfig["database"],
            valconfig["groundtruth"],
            valconfig["dataset_folder"],
        )

    def __len__(self):
        return self.n_query + self.n_database

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_folder, self.images[index])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, index

    def __repr__(self):
        return f"{self.path.split('/')[-1].rstrip('.pkl')}"

    def groundtruth(self):
        return self.gt
