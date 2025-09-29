from dataloaders import CorePlaceDataModule
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torchvision import transforms

def test_datamodule(dataconfig_path: str): 
    def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Denormalize a tensor image."""
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader = CorePlaceDataModule.from_dataconfig(
        dataconfig_path, 
        batch_size=10, 
        num_images_per_place=10
    )
    dataloader.setup()
    batch = next(iter(dataloader.train_dataloader()))

    # Organize images by class
    classes = defaultdict(list)
    for image, label in zip(batch[0], batch[1]):
        classes[label.item()].append(image)

    # Create visualization
    n_classes = len(classes)
    max_images_per_class = max(len(imgs) for imgs in classes.values())
    
    fig, axes = plt.subplots(
        n_classes, 
        max_images_per_class, 
        figsize=(max_images_per_class * 2, n_classes * 2)
    )
    
    # Handle single class case
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Images per Class', fontsize=16, fontweight='bold', y=0.995)
    
    for row_idx, (class_id, images) in enumerate(sorted(classes.items())):
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx] if n_classes > 1 else axes[col_idx]
            
            # Denormalize and display image
            img_denorm = denormalize(image)
            ax.imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
            ax.axis('off')
            
            # Add class label on first image of each row
            if col_idx == 0:
                ax.set_ylabel(f'Class {class_id}', fontsize=12, fontweight='bold', rotation=0, 
                             labelpad=40, va='center')
        
        # Hide empty subplots
        for col_idx in range(len(images), max_images_per_class):
            ax = axes[row_idx, col_idx] if n_classes > 1 else axes[col_idx]
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"Total images in batch: {len(batch[0])}")
    print(f"Number of classes: {n_classes}")
    print(f"\nImages per class:")
    for class_id, images in sorted(classes.items()):
        print(f"  Class {class_id}: {len(images)} images")
    print("="*50)

if __name__ == "__main__":
    