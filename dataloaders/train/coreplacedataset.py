import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


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
