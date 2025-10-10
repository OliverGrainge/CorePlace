import cv2
import numpy as np
from pipeline.base import CorePlaceStep
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial


# ===================== COLOR VARIANCE =====================
def compute_color_variance_from_path(image_path, image_size=256):
    """Compute color variance from image path"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.nan
        
        # Resize for consistency
        h, w = img.shape[:2]
        scale = image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Mean standard deviation across color channels
        color_variance = np.mean([np.std(img_small[:,:,i]) for i in range(3)])
        return color_variance
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.nan


class ColorVariance(CorePlaceStep):
    def __init__(self, threshold=20, image_size=256, n_jobs=-1):
        """
        Detect images with low color variance (uniform appearance).
        
        Args:
            threshold: minimum acceptable color variance
            image_size: resize parameter for consistency
            n_jobs: number of parallel jobs
        """
        self.threshold = threshold
        self.image_size = image_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_path, exist_ok=True)
    
    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        
        cache_key = self._compute_cache_key(dataconfig)
        cache_file = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_file):
            color_variances = np.load(cache_file)
            dataconfig["color_variance"] = color_variances
        else:
            image_paths = dataconfig["image_path"].values
            compute_fn = partial(compute_color_variance_from_path, image_size=self.image_size)
            
            if self.n_jobs == 1:
                color_variances = [compute_fn(p) for p in tqdm(image_paths, desc="Computing color variance")]
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    color_variances = list(
                        tqdm(
                            executor.map(compute_fn, image_paths),
                            total=len(image_paths),
                            desc="Computing color variance"
                        )
                    )
            
            dataconfig["color_variance"] = color_variances
            np.save(cache_file, dataconfig["color_variance"].values)
        
        pipe_state["dataconfig"] = dataconfig
        fig = self.plot(pipe_state)
        pipe_state["plots"].append({"figure": fig, "name": "color_variance"})
        return pipe_state
    
    def _compute_cache_key(self, dataconfig) -> str:
        image_paths = sorted(dataconfig["image_path"].tolist())
        cache_str = f"color_{self.image_size}|" + "|".join(image_paths)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_path, f"color_{cache_key}.npy")
    
    def plot(self, pipe_state: dict, N=5, quantiles=[0.0, 0.005, 0.01, 0.03, 0.05]):
        """Plot N images with lowest color variance above each quantile"""
        dataconfig = pipe_state["dataconfig"]
        color_variances = dataconfig["color_variance"].values
        image_paths = dataconfig["image_path"].values
        
        valid_mask = ~np.isnan(color_variances)
        color_variances = color_variances[valid_mask]
        image_paths = image_paths[valid_mask]
        
        fig, axes = plt.subplots(len(quantiles), N, figsize=(N * 3, len(quantiles) * 3))
        if len(quantiles) == 1:
            axes = axes.reshape(1, -1)
        if N == 1:
            axes = axes.reshape(-1, 1)
        
        for i, q in enumerate(quantiles):
            threshold = np.quantile(color_variances, q)
            mask = color_variances >= threshold
            filtered_variances = color_variances[mask]
            filtered_paths = image_paths[mask]
            
            sorted_indices = np.argsort(filtered_variances)[:N]
            selected_variances = filtered_variances[sorted_indices]
            selected_paths = filtered_paths[sorted_indices]
            
            for j in range(N):
                ax = axes[i, j]
                if j < len(selected_paths):
                    try:
                        img = Image.open(selected_paths[j]).resize((320, 320))
                        ax.imshow(img)
                        ax.set_title(f"Variance: {selected_variances[j]:.2f}", fontsize=10)
                    except:
                        ax.text(0.5, 0.5, "Error", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                
                ax.axis('off')
                if j == 0:
                    ax.text(-0.1, 0.5, f"{q*100:.2f}%\nquantile", 
                           fontsize=12, fontweight='bold', ha='right', va='center', 
                           transform=ax.transAxes)
        
        fig.suptitle("Lowest Color Variance (Uniform Color)", fontsize=14, y=0.995)
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        return f"ColorVariance(threshold={self.threshold}, image_size={self.image_size}, n_jobs={self.n_jobs})"