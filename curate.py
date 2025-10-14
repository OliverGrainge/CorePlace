import sys

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

import os

import pandas as pd

from pipeline import CorePlacePipeline
from utils import load_config


def save_dataconfig(pipe_state: dict, name: str, parent_dir: str = None):
    dataconfig = pipe_state["dataconfig"]
    
    # Create nested folder structure if parent_dir is provided
    if parent_dir:
        save_dir = f"registry/coreplacesets/{parent_dir}/{name}/"
    else:
        save_dir = f"registry/coreplacesets/{name}/"
    
    os.makedirs(save_dir, exist_ok=True)
    pd.to_pickle(dataconfig, f"{save_dir}dataconfig.pkl")
    print("\n\n  PLOTS \n")
    for i, plot_info in enumerate(pipe_state["plots"]):
        fig = plot_info["figure"]
        plot_name = plot_info.get("name", i)
        # Save the figure
        save_path = f"{save_dir}{plot_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close after saving
        print(f"    ✓ {plot_name}.png")
    print_summary(dataconfig, name, save_dir)


def print_summary(dataconfig: pd.DataFrame, name: str, save_dir: str = None):
    num_classes = dataconfig["class_id"].nunique()
    num_instances = len(dataconfig)
    avg_instances = dataconfig["class_id"].value_counts().mean()
    min_instances = dataconfig["class_id"].value_counts().min()

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY".center(60))
    print("=" * 60)
    print(f"\n  Dataset: {name}")
    print(f"\n  Classes:              {num_classes:,}")
    print(f"  Total Instances:      {num_instances:,}")
    print(f"  Avg per Class:        {avg_instances:.1f}")
    print(f"  Min per Class:        {min_instances:,}")
    
    if save_dir:
        # Remove trailing slash for display
        display_path = save_dir.rstrip('/')
        print(f"\n  Saved to: {display_path}dataconfig.pkl")
    else:
        print(f"\n  Saved to: registry/coreplacesets/{name}/dataconfig.pkl")
    print("\n" + "=" * 60 + "\n")


def main():
    if len(sys.argv) < 2:
        print("\n❌ Error: Missing configuration file")
        print("Usage: python script.py <config.yaml>\n")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    
    # Extract filename without extension
    name = config_path.split("/")[-1].split(".")[0]
    
    # Extract parent directory name
    path_parts = config_path.split("/")
    parent_dir = path_parts[-2] if len(path_parts) > 1 else None

    print("\n" + "=" * 60)
    print("  DATA PIPELINE".center(60))
    print("=" * 60)
    print(f"\n  Config: {name}")
    if parent_dir:
        print(f"  Parent Directory: {parent_dir}")
    print(f"\n  Starting pipeline...\n")

    pipeline = CorePlacePipeline.from_config(config)
    pipe_state = pipeline.run()
    save_dataconfig(pipe_state, name, parent_dir)


if __name__ == "__main__":
    main()
