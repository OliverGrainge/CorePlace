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


# ===================== BLUR DETECTION =====================
def compute_blur_from_path(image_path):
    """Compute Laplacian variance (blur metric) from image path"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.nan
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.nan


class BlurDetection(CorePlaceStep):
    def __init__(self, threshold=100, n_jobs=-1):
        """
        Detect blurry images using Laplacian variance.
        Lower values indicate more blur.
        
        Args:
            threshold: blur threshold (lower = more blurry)
            n_jobs: number of parallel jobs (-1 = all cores)
        """
        self.threshold = threshold
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_path, exist_ok=True)
    
    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        
        cache_key = self._compute_cache_key(dataconfig)
        cache_file = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_file):
            blur_scores = np.load(cache_file)
            dataconfig["blur_score"] = blur_scores
        else:
            image_paths = dataconfig["image_path"].values
            
            if self.n_jobs == 1:
                tqdm.pandas(desc="Computing blur scores")
                dataconfig["blur_score"] = dataconfig["image_path"].progress_apply(compute_blur_from_path)
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    blur_scores = list(
                        tqdm(
                            executor.map(compute_blur_from_path, image_paths),
                            total=len(image_paths),
                            desc="Computing blur scores"
                        )
                    )
                dataconfig["blur_score"] = blur_scores
            
            np.save(cache_file, dataconfig["blur_score"].values)
        
        pipe_state["dataconfig"] = dataconfig
        fig = self.plot(pipe_state)
        pipe_state["plots"].append({"figure": fig, "name": "blur_detection"})
        return pipe_state
    
    def _compute_cache_key(self, dataconfig) -> str:
        image_paths = sorted(dataconfig["image_path"].tolist())
        cache_str = "blur|" + "|".join(image_paths)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_path, f"blur_{cache_key}.npy")
    
    def plot(self, pipe_state: dict, N=5, quantiles=[0.0, 0.005, 0.01, 0.03, 0.05]):
        """Plot N most blurry images (lowest blur scores) above each quantile"""
        dataconfig = pipe_state["dataconfig"]
        blur_scores = dataconfig["blur_score"].values
        image_paths = dataconfig["image_path"].values
        
        valid_mask = ~np.isnan(blur_scores)
        blur_scores = blur_scores[valid_mask]
        image_paths = image_paths[valid_mask]
        
        fig, axes = plt.subplots(len(quantiles), N, figsize=(N * 3, len(quantiles) * 3))
        if len(quantiles) == 1:
            axes = axes.reshape(1, -1)
        if N == 1:
            axes = axes.reshape(-1, 1)
        
        for i, q in enumerate(quantiles):
            threshold = np.quantile(blur_scores, q)
            mask = blur_scores >= threshold
            filtered_scores = blur_scores[mask]
            filtered_paths = image_paths[mask]
            
            sorted_indices = np.argsort(filtered_scores)[:N]
            selected_scores = filtered_scores[sorted_indices]
            selected_paths = filtered_paths[sorted_indices]
            
            for j in range(N):
                ax = axes[i, j]
                if j < len(selected_paths):
                    try:
                        img = Image.open(selected_paths[j]).resize((320, 320))
                        ax.imshow(img)
                        ax.set_title(f"Score: {selected_scores[j]:.1f}", fontsize=10)
                    except:
                        ax.text(0.5, 0.5, "Error", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                
                ax.axis('off')
                if j == 0:
                    ax.text(-0.1, 0.5, f"{q*100:.2f}%\nquantile", 
                           fontsize=12, fontweight='bold', ha='right', va='center', 
                           transform=ax.transAxes)
        
        fig.suptitle("Most Blurry Images (Lowest Laplacian Variance)", fontsize=14, y=0.995)
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        return f"BlurDetection(threshold={self.threshold}, n_jobs={self.n_jobs})"
