from pipeline.base import CorePlaceStep
import pandas as pd
from typing import Dict, Any
from tqdm import tqdm
import numpy as np
from PIL import Image

class ConfusionMixer(CorePlaceStep):
    """
    Corrupt a fraction of labels by assigning the label of their hardest negatives.
    """

    def __init__(self, rate: float = 0.1):
        self.rate = float(rate)

    def run(self, pipe_state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = pipe_state["dataconfig"]

        # Sort by hardness to select which rows to corrupt
        candidates = df.sort_values(
            by="neg_max_hardness", ascending=False
        )[["class_id", "neg_max_hardness_idx"]]

        n_to_flip = int(len(df) * self.rate)
        if n_to_flip <= 0 or candidates.empty:
            pipe_state["dataconfig"] = df
            return pipe_state

        # Ensure each target row is used at most once
        uniq = candidates.drop_duplicates(
            subset="neg_max_hardness_idx", keep="first"
        ).head(n_to_flip)

        # Process swaps with progress bar
        swapped_pairs = []
        target_indices = []
        new_labels = []
        
        print(f"Swapping labels for {len(uniq)} samples...")
        for idx, row in tqdm(uniq.iterrows(), total=len(uniq), desc="Corrupting labels"):
            target_idx = row["neg_max_hardness_idx"]
            
            # Resolve the actual index
            if target_idx in df.index:
                actual_target = target_idx
            else:
                actual_target = df.index[int(target_idx)]
            
            swapped_pairs.append((idx, actual_target))
            target_indices.append(actual_target)
            new_labels.append(int(row["class_id"]))

        # Assign the new labels in batch (ensuring int dtype)
        df.loc[target_indices, "class_id"] = new_labels

        pipe_state["dataconfig"] = df
        
        # Add plot of swapped negative pairs
        if swapped_pairs:
            swapped_fig = self.plot_swapped_negatives(df, swapped_pairs)
            pipe_state["plots"].append({
                "figure": swapped_fig,
                "name": "swapped_negatives"
            })

        return pipe_state

    def plot_swapped_negatives(self, df: pd.DataFrame, swapped_pairs: list):
        """
        Plot image pairs from different percentiles of hardness that were originally 
        negatives but had their labels swapped to positives.
        """
        import matplotlib.pyplot as plt
        
        # Sample from different percentiles
        n_total = len(swapped_pairs)
        percentiles = [0, 10, 25, 50, 75, 90, 100]  # Show 7 different percentiles
        n_pairs = min(len(percentiles), n_total)
        
        if n_pairs == 0:
            return plt.figure()
        
        # Get indices at different percentiles
        indices_to_plot = []
        for p in percentiles[:n_pairs]:
            idx = int(n_total * p / 100)
            if idx >= n_total:
                idx = n_total - 1
            indices_to_plot.append(idx)
        
        fig, axes = plt.subplots(n_pairs, 2, figsize=(8, 3 * n_pairs))
        
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for i, pair_idx in enumerate(indices_to_plot):
            source_idx, target_idx = swapped_pairs[pair_idx]
            source_row = df.loc[source_idx]
            target_row = df.loc[target_idx]
            
            percentile = percentiles[i]
            
            # Plot source image
            axes[i, 0].imshow(Image.open(source_row["image_path"]).convert("RGB"))
            axes[i, 0].set_title(f"Source (p{percentile})\nClass: {target_row['class_id']}")
            axes[i, 0].axis("off")
            
            # Plot target image (now same class after swap)
            axes[i, 1].imshow(Image.open(target_row["image_path"]).convert("RGB"))
            axes[i, 1].set_title(f"Target (Swapped)\nClass: {target_row['class_id']}")
            axes[i, 1].axis("off")
        
        plt.suptitle(f"Swapped Negative Pairs Across Percentiles (now positives) - Rate: {self.rate}", 
                     fontsize=14, y=0.995)
        plt.tight_layout()
        
        return fig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rate={self.rate})"