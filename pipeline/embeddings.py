import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import hub
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

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


def get_desc_size(model_name: str):
    if model_name == "EigenPlaces":
        return 2048
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_model(model_name: str):
    if model_name == "EigenPlaces":
        model = hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )
        transform = get_transform(model_name)
        desc_size = get_desc_size(model_name)
        return model, transform, desc_size
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

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        dataset = EmbeddingDataset(dataconfig, self.transform)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

        self._prepare_model()

        # Create a temporary file for the embeddings

        embeddings_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache/embeddings.dat"
        )
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        embeddings = np.memmap(
            embeddings_path,
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
            embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(len(dataconfig), self.desc_size),
        )
        self.model = self.model.cpu()
        del images, indices, desc
        return pipe_state

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
