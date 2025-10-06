from typing import Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

from pipeline.base import CorePlaceStep


class Sampler(CorePlaceStep):
    """
    A flexible sampler that can filter data based on a field's values or randomly.
    
    Args:
        sampling_field: Field name to use for value-based sampling. If None, uses random sampling.
        direction: Either "high" or "low" to sample highest or lowest values respectively.
        num_classes: Number of classes to select.
        num_instances_per_class: Number of instances to select per class.
        num_instances: Total number of instances to select.
        min_instances_per_class: Minimum number of instances required per class (filters out classes with fewer).
        class_percentile: Tuple of (lower, upper) percentiles for class-level filtering.
        instance_percentile: Tuple of (lower, upper) percentiles for instance-level filtering.
        random_seed: Seed for deterministic random sampling (used when sampling_field is None).
    """
    
    def __init__(
        self,
        sampling_field: Union[str, None] = None,
        direction: str = "high",
        num_classes: Union[int, None] = None,
        num_instances_per_class: Union[int, None] = None,
        num_instances: Union[int, None] = None,
        min_instances_per_class: Union[int, None] = None,
        class_percentile: Union[Tuple[int, int], None] = None,
        instance_percentile: Union[Tuple[int, int], None] = None,
        random_seed: int = 42,
    ):
        self.sampling_field = sampling_field
        self.direction = direction
        self.num_classes = num_classes
        self.num_instances_per_class = num_instances_per_class
        self.num_instances = num_instances
        self.min_instances_per_class = min_instances_per_class
        self.class_percentile = class_percentile
        self.instance_percentile = instance_percentile
        self.random_seed = random_seed
        
        # Validate direction
        if direction not in ["high", "low"]:
            raise ValueError(f"direction must be 'high' or 'low', got '{direction}'")
    
    @property
    def is_random_sampling(self):
        """Check if random sampling should be used."""
        return self.sampling_field is None
    
    @property
    def ascending(self):
        """Determine sort order based on direction (low=ascending, high=descending)."""
        return self.direction == "low"

    def _filter_percentile_class(
        self, dataconfig: pd.DataFrame, percentile: Tuple[int, int]
    ):
        """Filter classes based on their total field value percentile or randomly."""
        lower_percentile, upper_percentile = percentile
        
        if self.is_random_sampling:
            # For random sampling, select random subset of classes
            classes = dataconfig["class_id"].unique()
            n_classes = len(classes)
            n_select = int(n_classes * (upper_percentile - lower_percentile) / 100)
            
            rng = np.random.RandomState(self.random_seed)
            selected_classes = rng.choice(classes, size=min(n_select, n_classes), replace=False)
            
            filtered = dataconfig[dataconfig["class_id"].isin(selected_classes)]
            return filtered

        # Calculate total field value per class
        class_values = dataconfig.groupby("class_id")[self.sampling_field].sum()

        # Calculate percentile thresholds on class-level values
        lower_threshold = class_values.quantile(lower_percentile / 100)
        upper_threshold = class_values.quantile(upper_percentile / 100)

        # Find classes within the percentile range
        classes_in_range = class_values[
            (class_values >= lower_threshold) & (class_values <= upper_threshold)
        ].index

        # Filter dataconfig to keep only instances from those classes
        filtered = dataconfig[dataconfig["class_id"].isin(classes_in_range)]

        return filtered

    def _filter_percentile_instance(
        self, dataconfig: pd.DataFrame, percentile: Tuple[int, int]
    ):
        """Filter instances based on their individual field value percentile or randomly."""
        lower_percentile, upper_percentile = percentile
        
        if self.is_random_sampling:
            # For random sampling, select random instances
            n_instances = len(dataconfig)
            n_select = int(n_instances * (upper_percentile - lower_percentile) / 100)
            
            filtered = dataconfig.sample(
                n=min(n_select, n_instances), 
                random_state=self.random_seed
            )
            return filtered

        lower_threshold = dataconfig[self.sampling_field].quantile(lower_percentile / 100)
        upper_threshold = dataconfig[self.sampling_field].quantile(upper_percentile / 100)

        filtered = dataconfig[
            (dataconfig[self.sampling_field] >= lower_threshold)
            & (dataconfig[self.sampling_field] <= upper_threshold)
        ]

        return filtered

    def _filter_num_classes(
        self, dataconfig: pd.DataFrame, num_classes: int
    ):
        """Select top N classes by field value sum or randomly."""
        if self.is_random_sampling:
            # Random sampling of classes
            classes = dataconfig["class_id"].unique()
            rng = np.random.RandomState(self.random_seed)
            selected_classes = rng.choice(classes, size=min(num_classes, len(classes)), replace=False)
            filtered = dataconfig[dataconfig["class_id"].isin(selected_classes)]
            return filtered
        
        class_values = dataconfig.groupby("class_id")[self.sampling_field].sum()
        class_values = class_values.sort_values(ascending=self.ascending)
        top_classes = class_values.head(num_classes).index
        filtered = dataconfig[dataconfig["class_id"].isin(top_classes)]
        return filtered

    def _filter_num_instances_per_class(
        self,
        dataconfig: pd.DataFrame,
        num_instances_per_class: int,
    ):
        """Select top N instances per class by field value or randomly."""
        n = max(int(num_instances_per_class), 0)

        if self.is_random_sampling:
            # Shuffle with consistent random state, then take top-k per class
            shuffled = dataconfig.sample(frac=1, random_state=self.random_seed)
            return shuffled.groupby("class_id", group_keys=False).head(n)

        # Deterministic: order then take top-k per class
        ordered = dataconfig.sort_values(by=self.sampling_field, ascending=self.ascending)
        return ordered.groupby("class_id", group_keys=False).head(n)

    def _filter_num_instances(
        self, dataconfig: pd.DataFrame, num_instances: int
    ):
        """Select top N instances by field value or randomly."""
        if self.is_random_sampling:
            # Random sampling of instances
            filtered = dataconfig.sample(
                n=min(num_instances, len(dataconfig)), 
                random_state=self.random_seed
            )
            return filtered
        
        dataconfig = dataconfig.sort_values(by=self.sampling_field, ascending=self.ascending)
        filtered = dataconfig.head(num_instances)
        return filtered

    def _filter_min_instances_per_class(
        self,
        dataconfig: pd.DataFrame,
        min_instances_per_class: int,
    ):
        """Filter out classes that have fewer than the minimum number of instances."""
        filtered = dataconfig.groupby("class_id", group_keys=False).filter(
            lambda x: len(x) >= min_instances_per_class
        )
        return filtered

    def _load_image_safe(self, image_path: str):
        """Safely load an image, return None if fails."""
        try:
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
        return None

    def _plot_class_grid(self, dataconfig: pd.DataFrame, num_classes: int = 5, images_per_class: int = 6):
        """
        Plot a grid with one class per row, showing sample images from each class.
        
        Args:
            dataconfig: DataFrame containing image paths and class IDs
            num_classes: Number of random classes to display
            images_per_class: Number of images to show per class
        """
        if "image_path" not in dataconfig.columns or "class_id" not in dataconfig.columns:
            return None
        
        # Sample random classes
        available_classes = dataconfig["class_id"].unique()
        num_classes = min(num_classes, len(available_classes))
        
        rng = np.random.RandomState(self.random_seed)
        selected_classes = rng.choice(available_classes, size=num_classes, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(num_classes, images_per_class, 
                                figsize=(images_per_class * 2, num_classes * 2))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Class-Based Sample Grid', fontsize=16, y=0.995)
        
        for class_idx, class_id in enumerate(selected_classes):
            class_data = dataconfig[dataconfig["class_id"] == class_id]
            
            # Sample images from this class
            sampled_images = class_data.sample(
                n=min(images_per_class, len(class_data)), 
                random_state=self.random_seed
            )
            
            for img_idx in range(images_per_class):
                ax = axes[class_idx, img_idx]
                
                if img_idx < len(sampled_images):
                    image_path = sampled_images.iloc[img_idx]["image_path"]
                    img = self._load_image_safe(image_path)
                    
                    if img is not None:
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, 'Image\nNot Found', 
                               ha='center', va='center', fontsize=8)
                else:
                    ax.axis('off')
                    continue
                
                ax.axis('off')
                
                # Add class label on first image of each row
                if img_idx == 0:
                    ax.set_ylabel(f'Class {class_id}', fontsize=10, rotation=0, 
                                 labelpad=40, ha='right', va='center')
        
        plt.tight_layout()
        return fig

    def _plot_random_grid(self, dataconfig: pd.DataFrame, num_images: int = 30, grid_cols: int = 6):
        """
        Plot a grid of randomly sampled images without class organization.
        
        Args:
            dataconfig: DataFrame containing image paths
            num_images: Total number of images to display
            grid_cols: Number of columns in the grid
        """
        if "image_path" not in dataconfig.columns:
            return None
        
        # Sample random images
        num_images = min(num_images, len(dataconfig))
        sampled_data = dataconfig.sample(n=num_images, random_state=self.random_seed)
        
        # Calculate grid dimensions
        grid_rows = int(np.ceil(num_images / grid_cols))
        
        # Create figure
        fig, axes = plt.subplots(grid_rows, grid_cols, 
                                figsize=(grid_cols * 2, grid_rows * 2))
        
        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Random Sample Grid', fontsize=16, y=0.995)
        
        for idx, (_, row) in enumerate(sampled_data.iterrows()):
            row_idx = idx // grid_cols
            col_idx = idx % grid_cols
            ax = axes[row_idx, col_idx]
            
            image_path = row["image_path"]
            img = self._load_image_safe(image_path)
            
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Found', 
                       ha='center', va='center', fontsize=8)
            
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_images, grid_rows * grid_cols):
            row_idx = idx // grid_cols
            col_idx = idx % grid_cols
            axes[row_idx, col_idx].axis('off')
        
        plt.tight_layout()
        return fig


    def plot(self, pipe_state: dict) -> dict:
        """
        Generate visualization plots for the sampled data.
        
        Creates three plots:
        1. Class-based grid: Random classes with images organized by row
        2. Random sample grid: Random images without class organization
        3. Class distribution: Bar chart showing number of instances per class
        
        The plots are added to pipe_state["plots"] as dictionaries with
        "figure" and "name" keys.
        """
        dataconfig = pipe_state["dataconfig"]
        
        # Initialize plots list if not present
        if "plots" not in pipe_state:
            pipe_state["plots"] = []
        
        # Plot 1: Class-based grid
        fig1 = self._plot_class_grid(dataconfig)
        if fig1 is not None:
            pipe_state["plots"].append({
                "figure": fig1,
                "name": "class_based_sample_grid"
            })
        
        # Plot 2: Random sample grid
        fig2 = self._plot_random_grid(dataconfig)
        if fig2 is not None:
            pipe_state["plots"].append({
                "figure": fig2,
                "name": "random_sample_grid"
            })
        
        
        return pipe_state

    def run(self, pipe_state: dict) -> dict:
        dataconfig = pipe_state["dataconfig"]
        dataconfig = dataconfig.reset_index(drop=True).sort_index()
        
        if self.min_instances_per_class is not None:
            dataconfig = self._filter_min_instances_per_class(
                dataconfig, self.min_instances_per_class
            )
        if self.class_percentile is not None:
            dataconfig = self._filter_percentile_class(
                dataconfig, self.class_percentile
            )
        if self.instance_percentile is not None:
            dataconfig = self._filter_percentile_instance(
                dataconfig, self.instance_percentile
            )
        if self.num_classes is not None:
            dataconfig = self._filter_num_classes(
                dataconfig, self.num_classes
            )
        if self.num_instances_per_class is not None:
            dataconfig = self._filter_num_instances_per_class(
                dataconfig, self.num_instances_per_class
            )
        if self.num_instances is not None:
            assert (
                self.num_instances_per_class is None
            ), "num_instances and num_instances_per_class cannot be set at the same time"
            dataconfig = self._filter_num_instances(
                dataconfig, self.num_instances
            )

        pipe_state["dataconfig"] = dataconfig
        
        # Add plotting function here
        pipe_state = self.plot(pipe_state)
        # It adds to pipe_state["plots"].append({"figure": figure, "name": name})
        
        return pipe_state

    def __repr__(self) -> str:
        return (
            f"ValueBasedSampler("
            f"sampling_field={self.sampling_field}, "
            f"direction={self.direction}, "
            f"num_classes={self.num_classes}, "
            f"num_instances_per_class={self.num_instances_per_class}, "
            f"num_instances={self.num_instances}, "
            f"class_percentile={self.class_percentile}, "
            f"instance_percentile={self.instance_percentile})"
        )