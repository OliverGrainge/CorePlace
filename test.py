from dataloaders.coreplacedataloader import CorePlaceDataModule
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import torch
import numpy as np

def load_config(config_path: str):
    config = OmegaConf.load(config_path)
    return config


if __name__ == "__main__":
    config = load_config("runs/train/midtrain.yaml")
    config.CorePlaceDataModule.dataconfig = "registry/coreplacesets/labelmix-percent[100]/dataconfig.pkl"
    dataloader = CorePlaceDataModule.from_config(config.CorePlaceDataModule)
    dataloader.setup()
    dl = dataloader.train_dataloader()
    batch = next(iter(dl))
    img_shape = batch[0].shape[1:]
    print(batch[0].reshape(-1, 4, *img_shape).shape)
    print(batch[1].shape)
    
    # Reshape images and get labels
    images = batch[0].reshape(-1, 4, *img_shape)  # Shape: (num_classes, 4, C, H, W)
    labels = batch[1]  # Shape: (num_classes,)
    
    # Sample a few classes (e.g., 5 classes)
    num_classes_to_plot = min(5, len(images))
    sampled_indices = np.random.choice(len(images), num_classes_to_plot, replace=False)
    
    # Create the plot
    fig, axes = plt.subplots(num_classes_to_plot, 4, figsize=(16, 4 * num_classes_to_plot))
    
    # Handle case with only 1 class
    if num_classes_to_plot == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, sample_idx in enumerate(sampled_indices):
        class_images = images[sample_idx]  # 4 images for this class
        class_label = labels[sample_idx].item()
        
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            
            # Convert image from (C, H, W) to (H, W, C) for plotting
            img = class_images[col_idx].cpu().numpy()
            
            # Handle different image formats
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
                # Normalize if needed (assuming values might be in [0, 1] or [-1, 1])
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            elif img.shape[0] == 1:  # Grayscale
                img = img[0]
                
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.axis('off')
            
            # Add class label to each image
            ax.set_title(f'Class {class_label}', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('batch_visualization_by_class.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'batch_visualization_by_class.png'")
    print(f"Visualized {num_classes_to_plot} classes with 4 images each")
    plt.close()