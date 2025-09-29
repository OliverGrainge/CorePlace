from typing import Union

import numpy as np
import torch

from .base import CorePlaceStep
import numpy as np
import torch
from typing import Union
from tqdm import tqdm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def cosine_similarity(
    a: np.ndarray, b: np.memmap, batch_size: int = 100, dtype=np.float16
):
    device = get_device()
    a_tensor = torch.from_numpy(a.copy()).to(device).float()
    sim = np.empty((a.shape[0], b.shape[0]), dtype=dtype)
    for i in range(0, b.shape[0], batch_size):
        end_idx = min(i + batch_size, b.shape[0])
        b_batch = torch.from_numpy(b[i:end_idx].copy()).to(device).float()
        sim_blk = a_tensor @ b_batch.mT
        sim[:, i:end_idx] = sim_blk.cpu().numpy().astype(dtype)
    return sim


def mshardness(
    sim_chunk: np.ndarray, batch_idx: int, labels: np.ndarray, margin: float = 0.1
):
    assert labels.shape[0] == sim_chunk.shape[1]
    n = sim_chunk.shape[0]
    pos_mask = labels[batch_idx : batch_idx + n, None] == labels[None, :]
    neg_mask = ~pos_mask
    eye_rows = np.arange(n)
    eye_cols = eye_rows + batch_idx
    pos_mask[eye_rows, eye_cols] = False
    neg_mask[eye_rows, eye_cols] = False
    hardness = np.zeros(n)
    for i in tqdm(range(sim_chunk.shape[0])):
        pos_idxs = np.where(pos_mask[i])
        neg_idxs = np.where(neg_mask[i])
        pos_sims = sim_chunk[i, pos_idxs[0]]
        neg_sims = sim_chunk[i, neg_idxs[0]]
        max_sim_neg = np.max(neg_sims)
        min_sim_pos = np.min(pos_sims)
        pos_hardness = np.maximum(max_sim_neg - (pos_sims - margin), 0)
        neg_hardness = np.maximum((neg_sims + margin) - min_sim_pos, 0)
        hardness[i] = pos_hardness.sum() + neg_hardness.sum()
    return hardness


class ComputeHardness(CorePlaceStep):
    def __init__(self, batch_size: int = 32, margin: float = 0.1):
        self.batch_size = batch_size
        self.margin = margin

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        embeddings = pipe_state["embeddings"]
        hardness = np.zeros(len(dataconfig))
        for batch_idx in range(0, len(dataconfig), self.batch_size):
            chunk_embeddings = embeddings[batch_idx : batch_idx + self.batch_size, :]
            sim_blk = cosine_similarity(
                chunk_embeddings, embeddings, batch_size=self.batch_size
            )
            hardness_blk = mshardness(
                sim_blk, batch_idx, dataconfig["class_id"].values, margin=self.margin
            )
            hardness[batch_idx : batch_idx + self.batch_size] = hardness_blk
        dataconfig["hardness"] = hardness
        pipe_state["dataconfig"] = dataconfig
        return pipe_state

    def __repr__(self) -> str:
        return f"Hardness()"
