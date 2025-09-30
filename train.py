"""Training script for CorePlace model using PyTorch Lightning."""

import sys
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import utils
from dataloaders import CorePlaceDataModule
from models import CorePlaceModel
from utils import load_config


def main() -> None:
    """Main training function."""
    # Parse config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    # Verify config file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    config = load_config(config_path)

    # Initialize components
    model = CorePlaceModel.from_config(config.CorePlaceModel)
    datamodule = CorePlaceDataModule.from_config(config.CorePlaceDataModule)

    # Setup logger (optional)
    logger = None
    if config.Logger == True:
        name = config_path.split("/")[-1].split(".")[0]
        logger = WandbLogger(project="CorePlace", name=name)

    # Setup callbacks (only add if present in config)
    callbacks: List[pl.Callback] = []

    if config.Checkpoint == True:
        name = config_path.split("/")[-1].split(".")[0]
        monitor = str(datamodule.val_datasets[0]) + "/recallat1"
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"checkpoints/{name}",
                monitor=monitor,
                filename=f"{name}" + "-{epoch}-{val_loss:.4f}",
                save_top_k=1,
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Initialize trainer
    trainer = pl.Trainer(
        **config.Trainer,
        logger=logger,
        callbacks=callbacks if callbacks else None,
    )

    # Train the model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
