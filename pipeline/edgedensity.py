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


# ===================== EDGE DENSITY =====================
def compute_edge_density_from_path(image_path, image_size=256):
    """Compute edge density from image path"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.nan
        
        # Resize for consistency
        h, w = img.shape[:2]
        scale = image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.nan


class EdgeDensity(CorePlaceStep):
    def __init__(self, threshold=0.05, image_size=256, n_jobs=-1):
        """
        Detect images with low edge density (uniform/textureless).
        
        Args:
            threshold: minimum acceptable edge density
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
            edge_densities = np.load(cache_file)
            dataconfig["edge_density"] = edge_densities
        else:
            image_paths = dataconfig["image_path"].values
            compute_fn = partial(compute_edge_density_from_path, image_size=self.image_size)
            
            if self.n_jobs == 1:
                edge_densities = [compute_fn(p) for p in tqdm(image_paths, desc="Computing edge density")]
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    edge_densities = list(
                        tqdm(
                            executor.map(compute_fn, image_paths),
                            total=len(image_paths),
                            desc="Computing edge density"
                        )
                    )
            
            dataconfig["edge_density"] = edge_densities
            np.save(cache_file, dataconfig["edge_density"].values)
        
        pipe_state["dataconfig"] = dataconfig
        fig = self.plot(pipe_state)
        pipe_state["plots"].append({"figure": fig, "name": "edge_density"})
        return pipe_state
    
    def _compute_cache_key(self, dataconfig) -> str:
        image_paths = sorted(dataconfig["image_path"].tolist())
        cache_str = f"edge_{self.image_size}|" + "|".join(image_paths)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_path, f"edge_{cache_key}.npy")
    
    def plot(self, pipe_state: dict, N=5, quantiles=[0.0, 0.005, 0.01, 0.03, 0.05]):
        """Plot N images with lowest edge density above each quantile"""
        dataconfig = pipe_state["dataconfig"]
        edge_densities = dataconfig["edge_density"].values
        image_paths = dataconfig["image_path"].values
        
        valid_mask = ~np.isnan(edge_densities)
        edge_densities = edge_densities[valid_mask]
        image_paths = image_paths[valid_mask]
        
        fig, axes = plt.subplots(len(quantiles), N, figsize=(N * 3, len(quantiles) * 3))
        if len(quantiles) == 1:
            axes = axes.reshape(1, -1)
        if N == 1:
            axes = axes.reshape(-1, 1)
        
        for i, q in enumerate(quantiles):
            threshold = np.quantile(edge_densities, q)
            mask = edge_densities >= threshold
            filtered_densities = edge_densities[mask]
            filtered_paths = image_paths[mask]
            
            sorted_indices = np.argsort(filtered_densities)[:N]
            selected_densities = filtered_densities[sorted_indices]
            selected_paths = filtered_paths[sorted_indices]
            
            for j in range(N):
                ax = axes[i, j]
                if j < len(selected_paths):
                    try:
                        img = Image.open(selected_paths[j]).resize((320, 320))
                        ax.imshow(img)
                        ax.set_title(f"Density: {selected_densities[j]:.4f}", fontsize=10)
                    except:
                        ax.text(0.5, 0.5, "Error", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                
                ax.axis('off')
                if j == 0:
                    ax.text(-0.1, 0.5, f"{q*100:.2f}%\nquantile", 
                           fontsize=12, fontweight='bold', ha='right', va='center', 
                           transform=ax.transAxes)
        
        fig.suptitle("Lowest Edge Density (Textureless Images)", fontsize=14, y=0.995)
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        return f"EdgeDensity(threshold={self.threshold}, image_size={self.image_size}, n_jobs={self.n_jobs})"
