import contextlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import hub
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import hashlib

from .base import CorePlaceStep


def get_transform(model_name: str):
    if model_name == "EigenPlaces":
        transform = transforms.Compose(
            [
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform
    elif model_name == "MegaLoc":
        transform = transforms.Compose([
            transforms.Resize(256),                    # Resize shorter side to 256
            transforms.CenterCrop(224),                # or 518 for larger models
            transforms.ToTensor(),                     # [0, 255] → [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_desc_size(model_name: str):
    if model_name == "EigenPlaces":
        return 2048
    elif model_name == "MegaLoc":
        return 8448
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_model(model_name: str):
    if model_name == "EigenPlaces":
        # Suppress stdout and stderr during model loading
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
            devnull
        ), contextlib.redirect_stderr(devnull):
            model = hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            )
        transform = get_transform(model_name)
        desc_size = get_desc_size(model_name)
        return model, transform, desc_size
    elif model_name == "MegaLoc": 
        model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        return model, get_transform(model_name), get_desc_size(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


class EmbeddingDataset(Dataset):
    def __init__(self, dataconfig: pd.DataFrame, transform: transforms.Compose):
        self.dataconfig = dataconfig
        self.transform = transform

    def __len__(self):
        return len(self.dataconfig)

    def __getitem__(self, index):
        row = self.dataconfig.iloc[index]
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, index  # Return the positional index, not image_id


class Embeddings(CorePlaceStep):
    def __init__(self, model_name: str, batch_size: int = 32, num_workers: int = 0):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model, self.transform, self.desc_size = get_model(model_name)
        self._tempfile = None  # Will hold the NamedTemporaryFile object
        self.device = self._get_device()

        self.cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_path, exist_ok=True)

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]

        # Check cache first
        cache_key = self._compute_cache_key(dataconfig)
        cache_path = self._get_cache_path(cache_key)

        if os.path.exists(cache_path):
            pipe_state["embeddings"] = np.memmap(
                cache_path,
                dtype=np.float16,
                mode="r",
                shape=(len(dataconfig), self.desc_size),
            )
            return pipe_state

       # Otherwise compute embeddings (existing code)
        dataset = EmbeddingDataset(dataconfig, self.transform)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        
        self._prepare_model()
        
        embeddings = np.memmap(
            cache_path,  # Use cache_path instead of fixed path
            dtype=np.float16,
            mode="w+",
            shape=(len(dataconfig), self.desc_size),
        )
        
        self.model = self.model.to(self.device)
        for batch in tqdm(dataloader, desc="Embedding images"):
            images, indices = batch
            images = images.to(self.device)
            with torch.no_grad():
                desc = self.model(images).cpu().detach().numpy().astype(np.float16)
                embeddings[indices.numpy()] = desc
        
        embeddings.flush()
        pipe_state["embeddings"] = np.memmap(
            cache_path,
            dtype=np.float16,
            mode="r",
            shape=(len(dataconfig), self.desc_size),
        )
        self.model = self.model.cpu()
        del images, indices, desc
        return pipe_state

    def _compute_cache_key(self, dataconfig: pd.DataFrame) -> str:
        """Compute a hash of the image paths to use as cache key"""
        # Sort to ensure consistent hash regardless of order
        image_paths = sorted(dataconfig["image_path"].tolist())
        paths_str = "|".join(image_paths)
        # Include model_name to invalidate cache if model changes
        cache_str = f"{self.model_name}|{paths_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()


    def _get_cache_path(self, cache_key: str) -> str:
        """Get the path for cached embeddings"""
        cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"embeddings_{cache_key}.dat")


    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _prepare_model(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def __repr__(self) -> str:
        return f"Embeddings(model_name={self.model_name}, device={self.device})"
