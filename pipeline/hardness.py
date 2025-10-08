from typing import Union

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd 
import hashlib
import os

from .base import CorePlaceStep

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    pos_sum_hardness = np.zeros(n)
    neg_sum_hardness = np.zeros(n)
    pos_max_hardness = np.zeros(n)
    neg_max_hardness = np.zeros(n)
    pos_max_hardness_idx = np.zeros(n)
    neg_max_hardness_idx = np.zeros(n)
    for i in range(sim_chunk.shape[0]):
        pos_idxs = np.where(pos_mask[i])
        neg_idxs = np.where(neg_mask[i])
        pos_sims = sim_chunk[i, pos_idxs[0]]
        neg_sims = sim_chunk[i, neg_idxs[0]]

        # Handle case where there are no positive or negative samples
        if len(pos_sims) == 0 or len(neg_sims) == 0:
            pos_sum_hardness[i] = 0.0
            neg_sum_hardness[i] = 0.0
            pos_max_hardness[i] = 0.0
            neg_max_hardness[i] = 0.0
            continue

        max_sim_neg = np.max(neg_sims)
        min_sim_pos = np.min(pos_sims)
        pos_hardness = np.maximum(max_sim_neg - (pos_sims - margin), 0)
        neg_hardness = np.maximum((neg_sims + margin) - min_sim_pos, 0)
        pos_sum_hardness[i] = pos_hardness.sum()
        neg_sum_hardness[i] = neg_hardness.sum()
        pos_max_hardness[i] = np.max(pos_hardness)
        neg_max_hardness[i] = np.max(neg_hardness)
        pos_max_hardness_idx[i] = pos_idxs[0][np.argmax(pos_hardness)]
        neg_max_hardness_idx[i] = neg_idxs[0][np.argmax(neg_hardness)]
    return (
        pos_sum_hardness,
        neg_sum_hardness,
        pos_max_hardness,
        neg_max_hardness,
        pos_max_hardness_idx,
        neg_max_hardness_idx,
    )


class MultiSimilarityHardness(CorePlaceStep):
    def __init__(self, batch_size: int = 32, margin: float = 0.1):
        self.batch_size = batch_size
        self.margin = margin
        
        # Set up cache directory
        self.cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"
        )
        os.makedirs(self.cache_path, exist_ok=True)

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        embeddings = pipe_state["embeddings"]
        
        # Check cache first
        cache_key = self._compute_cache_key(dataconfig)
        cache_file = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_file):
            # Load from cache
            cached_data = np.load(cache_file)
            dataconfig["pos_sum_hardness"] = cached_data["pos_sum_hardness"]
            dataconfig["neg_sum_hardness"] = cached_data["neg_sum_hardness"]
            dataconfig["pos_max_hardness"] = cached_data["pos_max_hardness"]
            dataconfig["neg_max_hardness"] = cached_data["neg_max_hardness"]
            dataconfig["sum_hardness"] = cached_data["pos_sum_hardness"] + cached_data["neg_sum_hardness"]
            dataconfig["pos_max_hardness_idx"] = cached_data["pos_max_hardness_idx"]
            dataconfig["neg_max_hardness_idx"] = cached_data["neg_max_hardness_idx"]
            
            # Compute derived metrics
            dataconfig["max_hardness"] = (
                dataconfig["pos_max_hardness"] + dataconfig["neg_max_hardness"]
            )
            dataconfig["mean_hardness"] = (
                dataconfig["pos_sum_hardness"] + dataconfig["neg_sum_hardness"]
            ) / 2
            dataconfig["pos_sum_hardness_mean"] = dataconfig["pos_sum_hardness"].mean()
            
            pipe_state["dataconfig"] = dataconfig
            pipe_state = self.plot(pipe_state)
            return pipe_state
        
        # Otherwise compute hardness (existing code)
        pos_sum_hardness = np.zeros(len(dataconfig))
        neg_sum_hardness = np.zeros(len(dataconfig))
        pos_max_hardness = np.zeros(len(dataconfig))
        neg_max_hardness = np.zeros(len(dataconfig))
        pos_max_hardness_idx = np.zeros(len(dataconfig))
        neg_max_hardness_idx = np.zeros(len(dataconfig))
        
        for batch_idx in tqdm(
            range(0, len(dataconfig), self.batch_size), desc="Computing hardness"
        ):
            chunk_embeddings = embeddings[batch_idx : batch_idx + self.batch_size, :]
            sim_blk = cosine_similarity(
                chunk_embeddings, embeddings, batch_size=self.batch_size
            )
            (
                pos_sum_hardness_blk,
                neg_sum_hardness_blk,
                pos_max_hardness_blk,
                neg_max_hardness_blk,
                pos_max_hardness_idx_blk,
                neg_max_hardness_idx_blk,
            ) = mshardness(
                sim_blk, batch_idx, dataconfig["class_id"].values, margin=self.margin
            )
            pos_sum_hardness[batch_idx : batch_idx + self.batch_size] = (
                pos_sum_hardness_blk
            )
            neg_sum_hardness[batch_idx : batch_idx + self.batch_size] = (
                neg_sum_hardness_blk
            )
            pos_max_hardness[batch_idx : batch_idx + self.batch_size] = (
                pos_max_hardness_blk
            )
            neg_max_hardness[batch_idx : batch_idx + self.batch_size] = (
                neg_max_hardness_blk
            )
            pos_max_hardness_idx[batch_idx : batch_idx + self.batch_size] = (
                pos_max_hardness_idx_blk
            )
            neg_max_hardness_idx[batch_idx : batch_idx + self.batch_size] = (
                neg_max_hardness_idx_blk
            )
        
        # Save to cache
        np.savez_compressed(
            cache_file,
            pos_sum_hardness=pos_sum_hardness,
            neg_sum_hardness=neg_sum_hardness,
            pos_max_hardness=pos_max_hardness,
            neg_max_hardness=neg_max_hardness,
            pos_max_hardness_idx=pos_max_hardness_idx,
            neg_max_hardness_idx=neg_max_hardness_idx,
        )

        dataconfig["pos_sum_hardness"] = pos_sum_hardness
        dataconfig["neg_sum_hardness"] = neg_sum_hardness
        dataconfig["pos_max_hardness"] = pos_max_hardness
        dataconfig["neg_max_hardness"] = neg_max_hardness
        dataconfig["max_hardness"] = pos_max_hardness + neg_max_hardness
        dataconfig["sum_hardness"] = pos_sum_hardness + neg_sum_hardness
        dataconfig["mean_hardness"] = (pos_sum_hardness + neg_sum_hardness) / 2
        dataconfig["pos_sum_hardness_mean"] = pos_sum_hardness.mean()
        dataconfig["pos_max_hardness_idx"] = pos_max_hardness_idx
        dataconfig["neg_max_hardness_idx"] = neg_max_hardness_idx
        pipe_state["dataconfig"] = dataconfig
        pipe_state = self.plot(pipe_state)
        return pipe_state

    def _compute_cache_key(self, dataconfig: pd.DataFrame) -> str:
        """Compute a hash based on image paths, class labels, and margin"""
        # Sort by image path to ensure consistent ordering
        sorted_data = dataconfig.sort_values("image_path").reset_index(drop=True)
        image_paths = sorted_data["image_path"].tolist()
        class_ids = sorted_data["class_id"].tolist()
        
        # Create a string combining all relevant factors
        paths_str = "|".join(image_paths)
        labels_str = "|".join(map(str, class_ids))
        cache_str = f"{paths_str}|{labels_str}|margin={self.margin}"
        
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the path for cached hardness results"""
        return os.path.join(self.cache_path, f"hardness_{cache_key}.npz")

    def plot(self, pipe_state: dict): 
        hardest_pos_fig = self.plot_hardest_pos(pipe_state["dataconfig"])
        pipe_state["plots"].append({
            "figure": hardest_pos_fig,
            "name": "hardest_positives"
        })
        hardest_neg_fig = self.plot_hardest_neg(pipe_state["dataconfig"])
        pipe_state["plots"].append({
            "figure": hardest_neg_fig,
            "name": "hardest_negatives"
        })
        return pipe_state

    def plot_hardest_pos(self, dataconfig: pd.DataFrame, N: int = 8):
        hardest_pos = dataconfig.sort_values(by="pos_max_hardness", ascending=False).head(N)
        hardest_anchor_paths = hardest_pos["image_path"].tolist()
        
        # Get positive image paths using the indices
        pos_indices = hardest_pos["pos_max_hardness_idx"].values
        hardest_pos_paths = dataconfig.iloc[pos_indices]["image_path"].tolist()
        
        # Calculate grid dimensions: 4 pairs per column
        pairs_per_column = 4
        n_columns = (N + pairs_per_column - 1) // pairs_per_column  # Ceiling division
        n_rows = min(N, pairs_per_column)
        
        # Create figure with subplots (each pair needs 2 subplot columns: anchor + positive)
        fig, axes = plt.subplots(n_rows, n_columns * 2, 
                                figsize=(5 * n_columns * 2, 4 * n_rows))
        
        # Ensure axes is 2D array for consistent indexing
        if N == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_columns == 1:
            axes = axes.reshape(-1, 2)
        
        # Plot each pair
        for i in range(N):
            col_group = i // pairs_per_column  # Which column group (0, 1, 2, ...)
            row = i % pairs_per_column  # Which row within that column group
            
            # Calculate subplot positions
            anchor_col = col_group * 2
            pos_col = col_group * 2 + 1
            
            # Load and plot anchor image
            try:
                anchor_img = Image.open(hardest_anchor_paths[i])
                axes[row, anchor_col].imshow(anchor_img)
                axes[row, anchor_col].set_title(f"Pair {i+1}: Anchor", fontsize=10)
                axes[row, anchor_col].axis('off')
            except Exception as e:
                axes[row, anchor_col].text(0.5, 0.5, f"Error loading\nanchor {i+1}", 
                                        ha='center', va='center')
                axes[row, anchor_col].axis('off')
            
            # Load and plot positive image
            try:
                pos_img = Image.open(hardest_pos_paths[i])
                axes[row, pos_col].imshow(pos_img)
                axes[row, pos_col].set_title(f"Pair {i+1}: Positive", fontsize=10)
                axes[row, pos_col].axis('off')
            except Exception as e:
                axes[row, pos_col].text(0.5, 0.5, f"Error loading\npositive {i+1}", 
                                        ha='center', va='center')
                axes[row, pos_col].axis('off')
        
        # Hide empty subplots if N is not a multiple of pairs_per_column
        total_slots = n_rows * n_columns
        for i in range(N, total_slots):
            col_group = i // pairs_per_column
            row = i % pairs_per_column
            anchor_col = col_group * 2
            pos_col = col_group * 2 + 1
            axes[row, anchor_col].axis('off')
            axes[row, pos_col].axis('off')
        
        plt.suptitle(f"Top {N} Hardest Positive Pairs", fontsize=14, y=0.995)
        plt.tight_layout()
        
        return fig

    def plot_hardest_neg(self, dataconfig: pd.DataFrame, N: int = 8):
        hardest_neg = dataconfig.sort_values(by="neg_max_hardness", ascending=False).head(N)
        hardest_anchor_paths = hardest_neg["image_path"].tolist()
        
        # Get negative image paths using the indices
        neg_indices = hardest_neg["neg_max_hardness_idx"].values
        hardest_neg_paths = dataconfig.iloc[neg_indices]["image_path"].tolist()
        
        # Calculate grid dimensions: 4 pairs per column
        pairs_per_column = 4
        n_columns = (N + pairs_per_column - 1) // pairs_per_column  # Ceiling division
        n_rows = min(N, pairs_per_column)
        
        # Create figure with subplots (each pair needs 2 subplot columns: anchor + negative)
        fig, axes = plt.subplots(n_rows, n_columns * 2, 
                                figsize=(5 * n_columns * 2, 4 * n_rows))
        
        # Ensure axes is 2D array for consistent indexing
        if N == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_columns == 1:
            axes = axes.reshape(-1, 2)
        
        # Plot each pair
        for i in range(N):
            col_group = i // pairs_per_column  # Which column group (0, 1, 2, ...)
            row = i % pairs_per_column  # Which row within that column group
            
            # Calculate subplot positions
            anchor_col = col_group * 2
            neg_col = col_group * 2 + 1
            
            # Load and plot anchor image
            try:
                anchor_img = Image.open(hardest_anchor_paths[i])
                axes[row, anchor_col].imshow(anchor_img)
                axes[row, anchor_col].set_title(f"Pair {i+1}: Anchor", fontsize=10)
                axes[row, anchor_col].axis('off')
            except Exception as e:
                axes[row, anchor_col].text(0.5, 0.5, f"Error loading\nanchor {i+1}", 
                                        ha='center', va='center')
                axes[row, anchor_col].axis('off')
            
            # Load and plot negative image
            try:
                neg_img = Image.open(hardest_neg_paths[i])
                axes[row, neg_col].imshow(neg_img)
                axes[row, neg_col].set_title(f"Pair {i+1}: Negative", fontsize=10)
                axes[row, neg_col].axis('off')
            except Exception as e:
                axes[row, neg_col].text(0.5, 0.5, f"Error loading\nnegative {i+1}", 
                                        ha='center', va='center')
                axes[row, neg_col].axis('off')
        
        # Hide empty subplots if N is not a multiple of pairs_per_column
        total_slots = n_rows * n_columns
        for i in range(N, total_slots):
            col_group = i // pairs_per_column
            row = i % pairs_per_column
            anchor_col = col_group * 2
            neg_col = col_group * 2 + 1
            axes[row, anchor_col].axis('off')
            axes[row, neg_col].axis('off')
        
        plt.suptitle(f"Top {N} Hardest Negative Pairs", fontsize=14, y=0.995)
        plt.tight_layout()
        return fig
        

    def __repr__(self) -> str:
        return f"MultiSimilarityHardness(margin={self.margin})"