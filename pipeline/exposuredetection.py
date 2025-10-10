from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pipeline.base import CorePlaceStep
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import hashlib




# ===================== EXPOSURE ISSUES =====================
def compute_exposure_from_path(image_path, clip_threshold=0.05):
    """Compute exposure metrics from image path"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"score": np.nan, "underexposed": False, "overexposed": False, "dynamic_range": np.nan}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        total_pixels = gray.size
        
        underexposed = hist[0:10].sum() / total_pixels > clip_threshold
        overexposed = hist[246:256].sum() / total_pixels > clip_threshold
        dynamic_range = int(gray.max()) - int(gray.min())
        
        # Composite score: higher = worse exposure
        score = (hist[0:10].sum() + hist[246:256].sum()) / total_pixels
        
        return {
            "score": score,
            "underexposed": underexposed,
            "overexposed": overexposed,
            "dynamic_range": dynamic_range
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"score": np.nan, "underexposed": False, "overexposed": False, "dynamic_range": np.nan}


class ExposureDetection(CorePlaceStep):
    def __init__(self, clip_threshold=0.05, n_jobs=-1):
        """
        Detect over/underexposed images.
        
        Args:
            clip_threshold: fraction of pixels that can be clipped
            n_jobs: number of parallel jobs
        """
        self.clip_threshold = clip_threshold
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_path, exist_ok=True)
    
    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        
        cache_key = self._compute_cache_key(dataconfig)
        cache_file = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_file):
            cached_data = np.load(cache_file, allow_pickle=True).item()
            dataconfig["exposure_score"] = cached_data["scores"]
            dataconfig["dynamic_range"] = cached_data["dynamic_range"]
        else:
            image_paths = dataconfig["image_path"].values
            compute_fn = partial(compute_exposure_from_path, clip_threshold=self.clip_threshold)
            
            if self.n_jobs == 1:
                results = [compute_fn(p) for p in tqdm(image_paths, desc="Computing exposure")]
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(
                        tqdm(
                            executor.map(compute_fn, image_paths),
                            total=len(image_paths),
                            desc="Computing exposure"
                        )
                    )
            
            dataconfig["exposure_score"] = [r["score"] for r in results]
            dataconfig["dynamic_range"] = [r["dynamic_range"] for r in results]
            
            np.save(cache_file, {
                "scores": dataconfig["exposure_score"].values,
                "dynamic_range": dataconfig["dynamic_range"].values
            })
        
        pipe_state["dataconfig"] = dataconfig
        fig = self.plot(pipe_state)
        pipe_state["plots"].append({"figure": fig, "name": "exposure_detection"})
        return pipe_state
    
    def _compute_cache_key(self, dataconfig) -> str:
        image_paths = sorted(dataconfig["image_path"].tolist())
        cache_str = f"exposure_{self.clip_threshold}|" + "|".join(image_paths)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_path, f"exposure_{cache_key}.npy")
    
    def plot(self, pipe_state: dict, N=5, quantiles=[0.95, 0.97, 0.99, 0.995, 1.0]):
        """Plot N worst exposure images (highest scores) below each quantile"""
        dataconfig = pipe_state["dataconfig"]
        exposure_scores = dataconfig["exposure_score"].values
        image_paths = dataconfig["image_path"].values
        
        valid_mask = ~np.isnan(exposure_scores)
        exposure_scores = exposure_scores[valid_mask]
        image_paths = image_paths[valid_mask]
        
        fig, axes = plt.subplots(len(quantiles), N, figsize=(N * 3, len(quantiles) * 3))
        if len(quantiles) == 1:
            axes = axes.reshape(1, -1)
        if N == 1:
            axes = axes.reshape(-1, 1)
        
        for i, q in enumerate(quantiles):
            threshold = np.quantile(exposure_scores, q)
            mask = exposure_scores <= threshold
            filtered_scores = exposure_scores[mask]
            filtered_paths = image_paths[mask]
            
            sorted_indices = np.argsort(filtered_scores)[-N:][::-1]
            selected_scores = filtered_scores[sorted_indices]
            selected_paths = filtered_paths[sorted_indices]
            
            for j in range(N):
                ax = axes[i, j]
                if j < len(selected_paths):
                    try:
                        img = Image.open(selected_paths[j]).resize((320, 320))
                        ax.imshow(img)
                        ax.set_title(f"Score: {selected_scores[j]:.3f}", fontsize=10)
                    except:
                        ax.text(0.5, 0.5, "Error", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                
                ax.axis('off')
                if j == 0:
                    ax.text(-0.1, 0.5, f"{q*100:.2f}%\nquantile", 
                           fontsize=12, fontweight='bold', ha='right', va='center', 
                           transform=ax.transAxes)
        
        fig.suptitle("Worst Exposure Issues (Over/Underexposed)", fontsize=14, y=0.995)
        plt.tight_layout()
        return fig
    
    def __repr__(self) -> str:
        return f"ExposureDetection(clip_threshold={self.clip_threshold}, n_jobs={self.n_jobs})"
