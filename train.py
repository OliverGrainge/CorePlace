"""Training script for CorePlace model using PyTorch Lightning."""

import argparse
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import utils
from dataloaders import CorePlaceDataModule
from models import CorePlaceModel
from utils import load_config

torch.set_float32_matmul_precision("medium")


def main() -> None:
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train CorePlace model")
    parser.add_argument("config", type=str, help="Path to base config file")
    parser.add_argument(
        "--dataconfig", type=str, required=True, help="Path to dataconfig pickle file"
    )
    args = parser.parse_args()

    # Verify config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Load configuration
    config = load_config(args.config)

    # Override dataconfig
    config.CorePlaceDataModule.dataconfig = args.dataconfig

    # Initialize components
    model = CorePlaceModel.from_config(config.CorePlaceModel)
    datamodule = CorePlaceDataModule.from_config(config.CorePlaceDataModule)

    # Setup logger (optional)
    logger = None
    if config.Logger == True:
        project_name = args.dataconfig.split("/")[-3].split(".")[0]
        experiment_name = args.dataconfig.split("/")[-2].split(".")[0]
        logger = WandbLogger(project=f"CorePlace-{project_name}", name=experiment_name)

    # Setup callbacks (only add if present in config)
    callbacks: List[pl.Callback] = []

    if config.Checkpoint == True:
        name = args.config.split("/")[-1].split(".")[0]
        monitor = str(datamodule.val_datasets[0]) + "/recallat1"
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"registry/checkpoints/{name}",
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
        log_every_n_steps=5,
        callbacks=callbacks if callbacks else None,
    )

    # Train the model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
