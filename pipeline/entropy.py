from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pipeline.base import CorePlaceStep
import os
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import hashlib

def get_entropy(img, image_size=256):
    # resize: preserve aspect ratio, long side = size
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # grayscale + entropy
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def compute_entropy_from_path(image_path, image_size=256):
    """Load image from path and compute entropy with error handling"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.nan
        return get_entropy(img, image_size=image_size)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.nan

class Entropy(CorePlaceStep): 
    def __init__(self, threshold=0.5, image_size=256, n_jobs=-1):
        """
        Args:
            threshold: entropy threshold (not used in current implementation)
            image_size: resize parameter for entropy calculation
            n_jobs: number of parallel jobs (-1 = all cores, 1 = no parallelization)
        """
        self.threshold = threshold
        self.image_size = image_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        # Set up cache directory
        self.cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(self.cache_path, exist_ok=True)
    
    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        
        # Check cache first
        cache_key = self._compute_cache_key(dataconfig)
        cache_file = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_file):
            # Load from cache
            entropies = np.load(cache_file)
            dataconfig["entropy"] = entropies
            pipe_state["dataconfig"] = dataconfig
            fig = self.plot(pipe_state)
            pipe_state["plots"].append({"figure": fig, "name": "entropy"})
            return pipe_state
        
        # Otherwise compute entropies
        image_paths = dataconfig["image_path"].values
        
        if self.n_jobs == 1:
            # Sequential processing with tqdm
            tqdm.pandas(desc="Computing entropy")
            dataconfig["entropy"] = dataconfig["image_path"].progress_apply(
                lambda x: compute_entropy_from_path(x, image_size=self.image_size)
            )
        else:
            # Parallel processing with progress bar
            compute_fn = partial(compute_entropy_from_path, image_size=self.image_size)
            
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                entropies = list(
                    tqdm(
                        executor.map(compute_fn, image_paths),
                        total=len(image_paths),
                        desc="Computing entropy"
                    )
                )
            dataconfig["entropy"] = entropies
        
        # Save to cache
        np.save(cache_file, dataconfig["entropy"].values)
        
        pipe_state["dataconfig"] = dataconfig
        fig = self.plot(pipe_state)
        pipe_state["plots"].append({"figure": fig, "name": "entropy"})
        return pipe_state

    def _compute_cache_key(self, dataconfig) -> str:
        """Compute a hash of the image paths and parameters to use as cache key"""
        # Sort to ensure consistent hash regardless of order
        image_paths = sorted(dataconfig["image_path"].tolist())
        paths_str = "|".join(image_paths)
        # Include image_size to invalidate cache if parameter changes
        cache_str = f"{self.image_size}|{paths_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the path for cached entropy values"""
        return os.path.join(self.cache_path, f"entropy_{cache_key}.npy")

    def plot(self, pipe_state: dict, N=5, quantiles=[0.0, 0.005, 0.01, 0.03, 0.05]):
        """
        Plot N lowest entropy images greater than or equal to each quantile threshold.
        
        Args:
            pipe_state: Pipeline state dictionary containing dataconfig
            N: Number of images to display per quantile
        
        Returns:
            matplotlib Figure object
        """
        dataconfig = pipe_state["dataconfig"]
        
        # Get entropy values and image paths
        entropies = dataconfig["entropy"].values
        image_paths = dataconfig["image_path"].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(entropies)
        entropies = entropies[valid_mask]
        image_paths = image_paths[valid_mask]
        
        # Define quantiles
        quantile_labels = [f"{q*100:.2f}%" for q in quantiles]
        
        # Create figure
        fig, axes = plt.subplots(len(quantiles), N, figsize=(N * 3, len(quantiles) * 3))
        
        # Ensure axes is 2D
        if len(quantiles) == 1:
            axes = axes.reshape(1, -1)
        if N == 1:
            axes = axes.reshape(-1, 1)
        
        # For each quantile
        for i, (q, q_label) in enumerate(zip(quantiles, quantile_labels)):
            # Calculate quantile threshold
            threshold = np.quantile(entropies, q)
            
            # Get images >= threshold
            mask = entropies >= threshold
            filtered_entropies = entropies[mask]
            filtered_paths = image_paths[mask]
            
            # Sort by entropy (lowest first) and take N
            sorted_indices = np.argsort(filtered_entropies)[:N]
            selected_entropies = filtered_entropies[sorted_indices]
            selected_paths = filtered_paths[sorted_indices]
            
            # Plot images
            for j in range(N):
                ax = axes[i, j]
                
                if j < len(selected_paths):
                    img_path = selected_paths[j]
                    entropy_val = selected_entropies[j]
                    
                    # Load and display image
                    try:
                        img = Image.open(img_path).resize((320, 320))
                        ax.imshow(img)
                        ax.set_title(f"{entropy_val:.2f}", fontsize=10)
                    except Exception as e:
                        ax.text(0.5, 0.5, "Error\nloading", 
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    # Not enough images above threshold
                    ax.text(0.5, 0.5, "N/A", 
                        ha='center', va='center', transform=ax.transAxes)
                
                ax.axis('off')
                
                # Add quantile label on the leftmost image
                if j == 0:
                    # Add text label outside the plot area
                    ax.text(-0.1, 0.5, f"{q_label}\nquantile", 
                        fontsize=12, fontweight='bold',
                        ha='right', va='center', 
                        transform=ax.transAxes)
        
        fig.suptitle("Lowest Entropy Images Above Each Quantile", fontsize=14, y=0.995)
        plt.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"Entropy(threshold={self.threshold}, image_size={self.image_size}, n_jobs={self.n_jobs})"