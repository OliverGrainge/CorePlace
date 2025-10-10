# CorePlace

A Visual Place Recognition (VPR) training framework with advanced data curation pipelines and experiment tracking.

## Overview

CorePlace is a modular framework for training visual place recognition models with a focus on intelligent dataset curation. The framework provides a two-stage workflow:

1. **Data Curation** (`curate.py`): Build curated datasets using composable pipeline steps
2. **Model Training** (`train.py`): Train models on curated datasets with experiment tracking

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Curation](#data-curation)
  - [Pipeline Steps](#pipeline-steps)
  - [Configuration Examples](#configuration-examples)
- [Model Training](#model-training)
- [Experiment Tracking](#experiment-tracking)
- [Project Structure](#project-structure)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CorePlace.git
cd CorePlace

# Install dependencies
pip install torch torchvision pytorch-lightning
pip install pandas numpy opencv-python pillow matplotlib tqdm
pip install wandb python-dotenv
```

## Quick Start

### 1. Curate a Dataset

Create a curation configuration file (e.g., `mycuration.yaml`):

```yaml
pipeline:
  - step: ReadGsvCities
    params:
      gsv_cities_path: /path/to/gsv-cities
      cities: 
        - London 
        - Boston
        - Tokyo

  - step: Entropy
    params: 
      image_size: 224

  - step: Sampler
    params:
      sampling_field: entropy
      min_instances_per_class: 6
      num_classes: 5000
      num_instances_per_class: 6
```

Run the curation pipeline:

```bash
python curate.py mycuration.yaml
```

This generates:
- `registry/coreplacesets/mycuration/dataconfig.pkl` - Curated dataset configuration
- `registry/coreplacesets/mycuration/*.png` - Visualization plots

### 2. Train a Model

Create a training configuration file (e.g., `mytrain.yaml`):

```yaml
CorePlaceModel: 
  arch_name: resnet50gem
  pretrained: True
  desc_dim: 1024
  learning_rate: 0.001
  weight_decay: 0.0001
  recallatks: [1, 5, 10, 25]

CorePlaceDataModule: 
  image_size: 320
  batch_size: 32
  num_workers: 8
  num_images_per_place: 4
  val_dataset_names: 
    - Pitts30k
    - MSLS

Trainer: 
  max_steps: 48000
  val_check_interval: 1000
  gradient_clip_val: 1.0

Logger: True
Checkpoint: True
```

Train the model:

```bash
python train.py mytrain.yaml --dataconfig registry/coreplacesets/mycuration/dataconfig.pkl
```

## Data Curation

The curation pipeline is designed as a sequence of modular steps that transform and filter your dataset. Each step receives a `pipe_state` dictionary containing a `dataconfig` DataFrame and can add visualizations to track the curation process.

### Pipeline Steps

#### 1. **ReadGsvCities**

Loads image data from Google Street View Cities dataset.

```yaml
- step: ReadGsvCities
  params:
    gsv_cities_path: /path/to/gsv-cities
    cities: 
      - London
      - Boston
      - Chicago
```

**Parameters:**
- `gsv_cities_path`: Path to GSV-Cities dataset
- `cities`: List of city names to include

---

#### 2. **Entropy**

Computes image entropy (information content) for each image. Lower entropy indicates simpler, potentially less informative images.

```yaml
- step: Entropy
  params: 
    image_size: 224
    n_jobs: -1
```

**Parameters:**
- `image_size`: Resize dimension for entropy calculation (default: 256)
- `n_jobs`: Number of parallel workers (-1 for all CPU cores)

**Adds field:** `entropy` (float)

**Visualization:** Shows lowest entropy images above different quantile thresholds

---

#### 3. **BlurDetection**

Detects blurry images using Laplacian variance. Lower scores indicate more blur.

```yaml
- step: BlurDetection
  params: 
    threshold: 100
    n_jobs: -1
```

**Parameters:**
- `threshold`: Blur threshold (images below are considered blurry)
- `n_jobs`: Number of parallel workers

**Adds field:** `blur_score` (float)

**Visualization:** Distribution of blur scores across the dataset

---

#### 4. **EdgeDensity**

Measures edge density in images using Canny edge detection.

```yaml
- step: EdgeDensity
  params: 
    n_jobs: -1
```

**Parameters:**
- `n_jobs`: Number of parallel workers

**Adds field:** `edge_density` (float)

**Visualization:** Edge density distribution

---

#### 5. **ColorVariance**

Computes color variance across RGB channels.

```yaml
- step: ColorVariance
  params: 
    n_jobs: -1
```

**Parameters:**
- `n_jobs`: Number of parallel workers

**Adds field:** `color_variance` (float)

**Visualization:** Color variance distribution

---

#### 6. **Embeddings**

Computes visual embeddings using a pretrained model.

```yaml
- step: Embeddings
  params:
    model_name: resnet50gem
    batch_size: 32
```

**Parameters:**
- `model_name`: Architecture name (e.g., resnet50gem)
- `batch_size`: Batch size for embedding computation

**Adds field:** `embedding` (stored in memmap for efficiency)

---

#### 7. **MultiSimilarityHardness**

Computes training hardness metrics based on embedding similarities.

```yaml
- step: MultiSimilarityHardness
  params:
    margin: 0.1
```

**Parameters:**
- `margin`: Margin for MultiSimilarity loss hardness

**Adds fields:**
- `pos_sum_hardness`: Sum of positive pair hardness
- `neg_sum_hardness`: Sum of negative pair hardness  
- `pos_max_hardness`: Maximum positive pair hardness
- `neg_max_hardness`: Maximum negative pair hardness
- `sum_hardness`: Total hardness (pos + neg)
- `max_hardness`: Maximum hardness

**Visualization:** Hardest positive and negative examples

---

#### 8. **Sampler**

Flexible sampling step for filtering and selecting data.

```yaml
- step: Sampler
  params:
    sampling_field: entropy              # Field to sample by (null for random)
    direction: low                        # "high" or "low"
    num_classes: 5000                    # Number of classes to keep
    num_instances_per_class: 6           # Instances per class
    min_instances_per_class: 4           # Filter out classes with fewer
    class_percentile: [0, 50]            # Keep bottom 50% of classes
    instance_percentile: [5, 100]        # Keep top 95% of instances
    random_seed: 42
```

**Parameters:**
- `sampling_field`: Field to use for value-based sampling (null for random sampling)
- `direction`: "high" for highest values, "low" for lowest values
- `num_classes`: Select top N classes
- `num_instances_per_class`: Select top N instances per class
- `num_instances`: Total instances to select (alternative to per-class)
- `min_instances_per_class`: Minimum instances required per class
- `class_percentile`: [lower, upper] percentile range for class-level filtering
- `instance_percentile`: [lower, upper] percentile range for instance-level filtering
- `random_seed`: Seed for deterministic random sampling

**Visualization:** 
- Class-based sample grid (one row per class)
- Random sample grid
- Class distribution

---

#### 9. **LabelMixer**

Randomly corrupts labels for a percentage of samples (for robustness testing).

```yaml
- step: LabelMixer 
  params: 
    rate: 0.1  # Corrupt 10% of labels
```

**Parameters:**
- `rate`: Percentage of samples to corrupt (0.0 to 1.0)

---

#### 10. **ConfusionMixer**

Swaps labels with visually similar but incorrect classes (based on embeddings).

```yaml
- step: ConfusionMixer
  params: 
    rate: 0.3  # Swap 30% of samples
```

**Parameters:**
- `rate`: Percentage of samples to swap with hard negatives

**Requires:** Embeddings step must be run first

**Visualization:** Shows examples of swapped negatives

---

### Configuration Examples

#### Example 1: Entropy-Based Curation

Curate a dataset by removing low-information (low entropy) images:

```yaml
pipeline:
  - step: ReadGsvCities
    params:
      gsv_cities_path: /path/to/gsv-cities
      cities: [London, Boston, Chicago]

  - step: Entropy
    params: 
      image_size: 224

  - step: Sampler
    params:
      sampling_field: entropy
      direction: high
      min_instances_per_class: 6
      instance_percentile: [5, 100]  # Remove bottom 5% entropy

  - step: Sampler  
    params:
      num_classes: 5000
      num_instances_per_class: 6
```

#### Example 2: Quality-Based Filtering

Remove blurry and low-variance images:

```yaml
pipeline:
  - step: ReadGsvCities
    params:
      gsv_cities_path: /path/to/gsv-cities
      cities: [London, Boston]

  - step: BlurDetection
    params: 
      threshold: 100

  - step: Sampler 
    params: 
      sampling_field: blur_score
      direction: high
      instance_percentile: [10, 100]  # Remove bottom 10% (most blurry)

  - step: ColorVariance

  - step: Sampler
    params:
      sampling_field: color_variance
      direction: high
      instance_percentile: [5, 100]  # Remove bottom 5% variance
```

#### Example 3: Hardness-Based Sampling

Sample based on training difficulty:

```yaml
pipeline:
  - step: ReadGsvCities
    params:
      gsv_cities_path: /path/to/gsv-cities
      cities: [London, Boston, Tokyo]

  - step: Embeddings
    params:
      model_name: resnet50gem
      batch_size: 32

  - step: MultiSimilarityHardness
    params:
      margin: 0.1

  - step: Sampler
    params:
      sampling_field: max_hardness
      direction: high
      min_instances_per_class: 6
      instance_percentile: [50, 100]  # Keep hardest 50%

  - step: Sampler
    params:
      num_classes: 7500
      num_instances_per_class: 7
```

#### Example 4: Label Noise Experiments

Test model robustness to label noise:

```yaml
pipeline:
  - step: ReadGsvCities
    params:
      gsv_cities_path: /path/to/gsv-cities
      cities: [London, Boston]

  - step: Sampler  
    params:
      min_instances_per_class: 4

  - step: LabelMixer 
    params: 
      rate: 0.1  # Corrupt 10% of labels

  - step: Sampler  
    params:
      min_instances_per_class: 4
      num_classes: 5000
```

#### Example 5: Confusion-Based Hard Negative Mining

```yaml
pipeline:
  - step: ReadGsvCities
    params:
      gsv_cities_path: /path/to/gsv-cities
      cities: [London, Boston]

  - step: Embeddings
    params:
      model_name: resnet50gem

  - step: Sampler
    params:
      min_instances_per_class: 4

  - step: ConfusionMixer
    params:
      rate: 0.3  # Replace 30% with hard negatives

  - step: Sampler
    params:
      min_instances_per_class: 4
```

## Model Training

### Training Configuration

The training configuration specifies the model architecture, data loading, and training parameters:

```yaml
CorePlaceModel: 
  arch_name: resnet50gem          # Model architecture
  pretrained: True                # Use pretrained weights
  desc_dim: 1024                  # Descriptor dimension
  learning_rate: 0.001            # Learning rate
  weight_decay: 0.0001            # Weight decay for regularization
  recallatks: [1, 5, 10, 25]     # Recall@K metrics to track

CorePlaceDataModule: 
  # dataconfig: specified via --dataconfig argument
  image_size: 320                 # Input image size
  batch_size: 32                  # Training batch size
  num_workers: 8                  # DataLoader workers
  num_images_per_place: 4         # Images per location in batch
  val_dataset_names:              # Validation datasets
    - Pitts30k
    - MSLS
    - Tokyo247

Trainer: 
  max_steps: 48000                # Maximum training steps
  val_check_interval: 1000        # Validation frequency
  check_val_every_n_epoch: null   # Alternative: validate every N epochs
  gradient_clip_val: 1.0          # Gradient clipping
  num_sanity_val_steps: 0         # Sanity validation steps

Logger: True                       # Enable Weights & Biases logging
Checkpoint: True                   # Save model checkpoints
```

### Training Command

```bash
python train.py <train_config.yaml> --dataconfig <path/to/dataconfig.pkl>
```

**Example:**

```bash
python train.py runs/train/midtrain.yaml \
  --dataconfig registry/coreplacesets/entropy-95-cls[5000]-inst[6]/dataconfig.pkl
```

### Validation Datasets

Configure validation datasets in `config.yaml`:

```yaml
val_datasets: 
  Amstertime: "/path/to/amstertime"
  Pitts30k: "/path/to/pitts30k"
  Tokyo247: "/path/to/tokyo247"
  Eynsham: "/path/to/eynsham"
  SVOX: "/path/to/svox"
  MSLS: "/path/to/msls"
```

The training script automatically evaluates on specified validation sets and reports Recall@K metrics.

## Experiment Tracking

### Data Pipeline Logging

Each curation run automatically generates:

1. **`dataconfig.pkl`**: The curated dataset configuration (Pandas DataFrame with image paths, labels, and computed features)

2. **Visualization Plots**: PNG files showing:
   - **Sample grids**: Visual inspection of selected images
   - **Feature distributions**: Histograms of computed metrics (entropy, blur, hardness, etc.)
   - **Quality analysis**: Per-step visualization of data transformations

**Example output structure:**

```
registry/coreplacesets/entropy-95-cls[5000]-inst[6]/
├── dataconfig.pkl              # Curated dataset
├── entropy.png                 # Entropy distribution visualization
├── class_based_sample_grid.png # Sample images organized by class
└── random_sample_grid.png      # Random sample of selected images
```

### Training Logging

#### Weights & Biases Integration

Set `Logger: True` in your training config to enable W&B logging:

```yaml
Logger: True
Checkpoint: True
```

**Logged metrics:**
- Training loss
- Learning rate schedule
- Validation Recall@K for each dataset
- Gradient norms
- System metrics (GPU usage, etc.)

**Setup:**

```bash
# Create .env file with your W&B API key
echo "WANDB_API_KEY=your_key_here" > .env

# Or set directly
export WANDB_API_KEY=your_key_here
```

The training script automatically:
- Creates a W&B run with the dataset name
- Logs metrics every 5 steps
- Uploads validation results at specified intervals

#### PyTorch Lightning Logs

Even without W&B, training logs are saved to `lightning_logs/`:

```
lightning_logs/
└── version_0/
    ├── events.out.tfevents.*  # TensorBoard logs
    ├── hparams.yaml           # Hyperparameters
    └── checkpoints/           # Model checkpoints
```

View with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

#### Model Checkpoints

When `Checkpoint: True`, models are saved to:

```
registry/checkpoints/<config_name>/
└── <config_name>-epoch=X-val_loss=Y.ckpt
```

Checkpoints are automatically saved for the best validation performance.

### Experiment Organization

The repository structure keeps experiments organized:

```
runs/
├── curate/
│   ├── entropy/
│   │   ├── entropy-95-cls[5000]-inst[6].yaml
│   │   └── entropy-95-cls[7500]-inst[7].yaml
│   ├── random/
│   │   └── random-cls[5000]-inst[6].yaml
│   └── hardness/
│       └── max-hardness-percentile[50-100]-cls[5000]-inst[6].yaml
└── train/
    ├── fasttrain.yaml
    ├── midtrain.yaml
    └── tinytrain.yaml

registry/
└── coreplacesets/
    ├── entropy-95-cls[5000]-inst[6]/
    │   ├── dataconfig.pkl
    │   └── *.png
    └── random-cls[5000]-inst[6]/
        ├── dataconfig.pkl
        └── *.png
```

### Running Experiments in Batch

Use the provided shell scripts:

```bash
# Curate multiple datasets
bash scripts/curate_all.sh

# Train on multiple configurations
bash scripts/train_all.sh
```

## Project Structure

```
CorePlace/
├── curate.py                    # Data curation script
├── train.py                     # Model training script
├── config.yaml                  # Global configuration (dataset paths)
│
├── pipeline/                    # Curation pipeline steps
│   ├── __init__.py
│   ├── base.py                 # Base class for pipeline steps
│   ├── read_gsvcities.py       # Load GSV-Cities dataset
│   ├── entropy.py              # Entropy computation
│   ├── blurdetection.py        # Blur detection
│   ├── edgedensity.py          # Edge density computation
│   ├── colourvariance.py       # Color variance
│   ├── embeddings.py           # Visual embeddings
│   ├── hardness.py             # Hardness metrics
│   ├── sampler.py              # Flexible sampling
│   ├── labelmixer.py           # Label noise injection
│   ├── confusionmixer.py       # Hard negative mining
│   └── cache/                  # Cached computations
│
├── models/                      # Model architectures
│   ├── archs.py                # Network architectures
│   └── model.py                # Lightning module
│
├── dataloaders/                 # Data loading
│   └── coreplacedataloader.py  # Dataset and DataModule
│
├── runs/                        # Experiment configurations
│   ├── curate/                 # Curation configs
│   │   ├── entropy/
│   │   ├── random/
│   │   ├── hardness/
│   │   ├── features/
│   │   ├── labelmixing/
│   │   └── confusionmixing/
│   └── train/                  # Training configs
│
├── registry/                    # Outputs
│   ├── coreplacesets/          # Curated datasets
│   └── checkpoints/            # Model checkpoints
│
├── scripts/                     # Utility scripts
│   ├── curate_all.sh
│   └── train_all.sh
│
└── utils.py                     # Utility functions
```

## Tips and Best Practices

### Data Curation

1. **Start Simple**: Begin with basic sampling, then add quality filters
2. **Inspect Visualizations**: Always check the generated plots to verify curation quality
3. **Cache Reuse**: Computed features (entropy, embeddings) are cached for reuse
4. **Iterative Refinement**: Chain multiple Sampler steps for complex filtering logic
5. **Balance Dataset**: Use `min_instances_per_class` to ensure class balance

### Training

1. **Validate on Multiple Datasets**: Use diverse validation sets to assess generalization
2. **Monitor Recall@1**: Primary metric for place recognition
3. **Checkpoint Best Models**: Keep `Checkpoint: True` to save best performing models
4. **Experiment Tracking**: Use descriptive configuration names for easy comparison

### Debugging

```bash
# Test curation pipeline with small dataset
python curate.py runs/curate/test.yaml

# Quick training test
python train.py runs/train/testtrain.yaml --dataconfig registry/coreplacesets/test/dataconfig.pkl
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{coreplace2024,
  title={CorePlace: Visual Place Recognition with Data Curation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgements

Built with [PyTorch Lightning](https://lightning.ai/) and [Weights & Biases](https://wandb.ai/).

---

For questions and support, please open an issue on GitHub.

